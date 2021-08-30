import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint #TensorBoard
from lstm_model import *
from keras.utils import plot_model
from helper_funcs import *
pd.options.mode.chained_assignment = None
from keras.models import load_model
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict
import csv
# from scipy import stats
import datetime
import pickle
np.set_printoptions(threshold=sys.maxsize)

def main():
    exp_args = ['test']
    args = sys.argv
    model_dir = 'experiments/' + args[1]
    epoch_to_predict = args[2]
    test_machine = args[3] # Machine to test the model, e.g TCV, AUG or JET
    # update dic
    exp_test_dic['machine_id'] = test_machine
    
    exp_train_dic = load_dic(model_dir + '/params_data_train')
    print('Settings for training: ', exp_train_dic)
    exp_test_dic = load_dic(model_dir + '/params_data_test')
    print('Settings for testing: ', exp_test_dic)

    c_offset = exp_train_dic['labelers']
    conv_window_size = exp_train_dic['conv_w_size']
    conv_w_offset = exp_train_dic['conv_w_offset']
    no_input_channels = exp_train_dic['no_input_channels']
    with_elms = exp_train_dic['with_elms'] 

    model_path = model_dir + '/model_checkpoints/weights.' + str(epoch_to_predict) + '.h5' 
    modelJoint = model_arc(bsize=1, conv_w_size=conv_window_size, no_input_channels=no_input_channels, timesteps=None, with_elms=with_elms)
    modelJoint.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    modelJoint.load_weights(model_path)
    #modelJoint.load_weights('./experiments/5/model_checkpoints/weights.100.h5')
    modelJoint.reset_states()
    modelJoint.summary()
    
    machine_id = exp_test_dic['machine_id']
   
    gaussian_time_window = 10e-4
    signal_sampling_rate = 1e4
    gaussian_hinterval = int(gaussian_time_window * signal_sampling_rate)
    print('Will count as correct ELM predictions within', gaussian_hinterval, 'time slices of ELM label')

    for exp_arg in exp_args:
        print('--------------------------------------------------------------------------------------------------' +
              str(exp_arg)+
              '--------------------------------------------------------------------------------------------------')

        exp_dic = load_dic(model_dir + '/params_data_' + exp_arg)
        
        # Update shots list based on tested machine
        if machine_id == 'JET':
            exp_dic['shot_ids'] = [97927, 97803, 96532, 97828, 97830, 97832, 97835, 97971, 97465, 97466, 97468, 97469, 97473, 94785, 97476, 97477, 96713, 97745, 98005, 96993, 96745, 94315, 97396, 96885, 97398, 97399, 94968, 94969, 94971, 94973]
        elif machine_id == 'AUG':
            exp_dic['shot_ids'] = [35248, 35274, 35275, 35529, 35530, 35531, 35532, 35536, 35538, 35540, 35556, 35557, 35564, 35582, 35584, 35604, 35607, 35837, 35852, 35967, 35972, 35975, 35537, 35539, 35561]
        
        labelers = exp_dic['labelers']
        c_offset = exp_dic['labelers']
        shots = [str(s) for s in exp_dic['shot_ids']]
        conv_window_size = exp_dic['conv_w_size']
        conv_w_offset = exp_dic['conv_w_offset']
        normalization = exp_dic['normalization']
        no_input_channels = exp_dic['no_input_channels']
        pred_args = [model_dir, epoch_to_predict, exp_arg, labelers, conv_w_offset, shots, conv_window_size, no_input_channels, gaussian_hinterval, machine_id, normalization, with_elms]
        predict(pred_args, modelJoint)
    
def predict(args, modelJoint):
    exp_arg = args[2]
    shots = args[5]
    labelers = args[3]
    gaussian_hinterval = args[8]
    
    data_dir = './labeled_data/'
    machine_id = args[9]
    normalization = args[10]
    with_elms = args[11]
    data_dir = './labeled_data/' + machine_id + '/'
    X_scalars_test = []
    fshots = {}
    conv_window_size = args[6]
    no_input_channels = args[7]
    conv_w_offset = int(args[4])
    intersect_times_d = {}
    for i, shot in zip(range(len(shots)), shots):
        print('Reading shot', shot)
        fshot = pd.read_csv(data_dir + labelers[0] + '/' + machine_id + '_'  + str(shot) + '_' + labelers[0] + '_labeled.csv', encoding='utf-8')
        shot_df = fshot.copy()
        if machine_id == 'JET':
            fshot = remove_current_750kA(shot_df)
        elif machine_id == 'TCV':
            fshot = remove_current_30kA(shot_df)
        elif machine_id == 'AUG':
            fshot = remove_current_150kA(shot_df)
        else:
            raise ValueError('Machine_id {} not stored in the database'.format(machine_id))
        shot_df = remove_no_state(shot_df)
        shot_df = remove_disruption_points(shot_df)
        shot_df = shot_df.reset_index(drop=True)
        shot_df = normalize_current_MA(shot_df)
        shot_df = normalize_signals_mean(shot_df, machine_id, func=normalization)
        
        intersect_times = np.round(shot_df.time.values,5)
        print('fshot shape: ', fshot.shape)
        if len(labelers) > 1:
            for k, labeler in enumerate(labelers):
                fshot_labeled = pd.read_csv(data_dir+ labeler + '/' + machine_id + '_'  + str(shot) + '_' + labeler + '_labeled.csv', encoding='utf-8')
                intersect_times = np.round(sorted(set(np.round(fshot_labeled.time.values,5)) & set(np.round(intersect_times,5))), 5)
        fshot_equalized = shot_df.loc[shot_df['time'].round(5).isin(intersect_times)]
        print('fshot_equalized shape: ', fshot_equalized.shape)
        intersect_times = intersect_times[conv_window_size-conv_w_offset:len(intersect_times)-conv_w_offset]
        intersect_times_d[shot] = intersect_times
        
        stride=10
        length = int(np.ceil((len(fshot_equalized)-conv_window_size)/stride))
        X_scalars_single = np.empty((length, conv_window_size, no_input_channels)) 
        for j in np.arange(length):
            vals = fshot_equalized.iloc[j*stride : conv_window_size + j*stride]
            #scalars = np.asarray([vals.FIR, vals.DML, vals.PD]).swapaxes(0, 1)
            scalars = np.asarray([vals.FIR, vals.PD, vals.DML]).swapaxes(0, 1)
            assert scalars.shape == (conv_window_size, no_input_channels)
            X_scalars_single[j] = scalars
        X_scalars_test += [X_scalars_single]
    # except:
    #     print('Shot', shot, 'does not exist in the database.')
        # exit(0)
    
    model_dir = args[0]
    epoch_to_predict = args[1]
    
    print('Predicting on shot(s)...')
    pred_states = []
    pred_elms = []
    pred_transitions =[]
    k_indexes =[]
    dice_cfs = []
    conf_mats = []
    conf_mets = []
    dice_cfs_dic = {}
    k_indexes_dic = {}
    pred_start = datetime.datetime.now()
    array_sdir = model_dir + '/epoch_' + epoch_to_predict + '/network_np_out_' + machine_id + '_Xkhz/' + exp_arg + '/'
    if not os.path.isdir(array_sdir):
        os.makedirs(array_sdir)
    for s_ind, s in enumerate(shots):
        print('Predicting shot ' + str(shots[s_ind]))
        modelJoint.reset_states()
        if with_elms:
            states, elms = modelJoint.predict(
                                    np.asarray([X_scalars_test[s_ind][:, :, :]]),
                                    batch_size=1,
                                    verbose=1)
        else: 
            states = modelJoint.predict(
                                    np.asarray([X_scalars_test[s_ind][:, :, :]]),
                                    batch_size=1,
                                    verbose=1)
        print('Predicted sequence length is', str(states.shape))
        np.save(array_sdir + str(s) + '_states_pred.npy', states)
        #np.save(array_sdir + str(s) + '_elms_pred.npy', elms)
       
         # sys.stdout.flush()
         # pred_transitions += [transitions]
    # pred_finish = datetime.datetime.now()
    # print('total prediction time: ', pred_finish - pred_start)
    # # CONVERT TO DISCRETE LABELS and save
    # 
    # 
    # 
    # for s_ind, s in enumerate(shots):
    #     np.save(array_sdir + str(s) + '_states_pred.npy', pred_states[s_ind])
    #     np.save(array_sdir + str(s) + '_elms_pred.npy', pred_elms[s_ind])
    #     
    #     

    
if __name__ == '__main__':
    main()

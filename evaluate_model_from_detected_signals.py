import sys
import os
import numpy as np
import pandas as pd
from lstm_model import *
from helper_funcs import *
pd.options.mode.chained_assignment = None
from collections import OrderedDict
import csv
import datetime
import pickle
import itertools
np.set_printoptions(threshold=sys.maxsize)

def main():
    exp_args = ['test']
    args = sys.argv
    # Get current directory path
    dirpath = os.getcwd()
    model_dir = os.path.join(dirpath, 'algorithms/ConvLSTM/experiments/' + args[1])
    epoch_to_predict = args[2]
    shot = args[3]
    # Machine from where the test shot belong
    machine_id = args[4]
    # Normalization choice (e.g minmax, avg, scaling, ...)
    normalization_method = args[5]
    # Set to true if it's a git test
    git_test = args[6]
    
    exp_train_dic = load_dic(model_dir + '/params_data_train')
    exp_test_dic = load_dic(model_dir + '/params_data_test')
    
    # If with_elms doesn't exist set it to the original value to test standalone baseline 16042021
    try:
        with_elms = exp_train_dic['with_elms']
    except KeyError:
        with_elms = True
    
    no_input_channels = exp_train_dic['no_input_channels']
    conv_w_size = exp_train_dic['conv_w_size']
    conv_w_offset = exp_train_dic['conv_w_offset']
    labeler = exp_train_dic['labelers']
    
    # Machine from where the model was trained
    train_machine = exp_train_dic['machine_id']
    postfix_name = labeler[0] + '_labeled'
    if eval(git_test):
        postfix_name = 'signals'

    model_path = model_dir + '/model_checkpoints/weights.' + str(epoch_to_predict) + '.h5'     
    convlstm = ConvLSTM(bsize=1, conv_w_size=conv_w_size, no_input_channels=no_input_channels, timesteps=None, with_elms=with_elms)
    modelJoint = convlstm.create_architecture()

    modelJoint.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    modelJoint.load_weights(model_path)
    modelJoint.reset_states()
    modelJoint.summary()
    
    gaussian_time_window = 10e-4
    signal_sampling_rate = 1e4
    gaussian_hinterval = int(gaussian_time_window * signal_sampling_rate)

    pred_args = [model_dir, epoch_to_predict, epoch_to_predict, postfix_name, conv_w_offset, shot, conv_w_size, no_input_channels, gaussian_hinterval, machine_id, train_machine, with_elms, git_test, normalization_method]
    predict(pred_args, modelJoint)
    
def predict(args, modelJoint):
    
    shot = args[5]
    postfix_name = args[3]
    gaussian_hinterval = args[8]
    # Get current directory path
    dirpath = os.getcwd()
    machine_id = args[9]
    train_machine = args[10]
    with_elms = args[11]
    git_test = args[12]
    normalization_method = args[13]
    if eval(git_test):
        data_dir = os.path.join(dirpath, 'data/Detected/')
    else:
        data_dir = os.path.join(dirpath, 'data/Validated/')
    X_scalars_test = []
    fshots = {}
    conv_window_size = args[6]
    no_input_channels = args[7]
    conv_w_offset = int(args[4])
    intersect_times_d = {}
    print('Reading shot', shot)
    print('from file: ', data_dir + machine_id + '_'  + str(shot) + '_' + postfix_name + '.csv')
    fshot = pd.read_csv(data_dir + machine_id + '_'  + str(shot) + '_' + postfix_name + '.csv', encoding='utf-8')
    shot_df = fshot.copy()
    if machine_id == 'JET':
        shot_df = remove_current_1MA(shot_df)
    elif machine_id == 'TCV':
        shot_df = remove_current_30kA(shot_df)
    elif machine_id == 'AUG':
        shot_df = remove_current_370kA(shot_df)
    else:
        raise ValueError('Machine_id {} not stored in the database'.format(machine_id))
    #shot_df = remove_no_state(shot_df)
    shot_df = replace_NaNs(shot_df)
    #shot_df = remove_disruption_points(shot_df)
    shot_df = shot_df.reset_index(drop=True)
    shot_df = normalize_current_MA(shot_df)
    shot_df = normalize_signals_mean(shot_df, machine_id, normalization_method)
    
    intersect_times = np.round(shot_df.time.values,5)
    fshot_labeled = pd.read_csv(data_dir+ machine_id + '_'  + str(shot) + '_' + postfix_name + '.csv', encoding='utf-8')
    intersect_times = np.round(sorted(set(np.round(fshot_labeled.time.values,5)) & set(np.round(intersect_times,5))), 5)
    
    fshot_equalized = shot_df.loc[shot_df['time'].round(5).isin(intersect_times)]
    intersect_times = intersect_times[conv_window_size-conv_w_offset:len(intersect_times)-conv_w_offset]
    
    stride=10
    length = int(np.ceil((len(fshot_equalized)-conv_window_size)/stride))
    X_scalars_single = np.empty((length, conv_window_size, no_input_channels)) # First LSTM predicted value will correspond to index 20 of the full sequence. 
    # fshots[shot] = fshot.ix[shot_df.index.values]
    for j in np.arange(length):
        vals = fshot_equalized.iloc[j*stride : conv_window_size + j*stride]
        if no_input_channels == 2: # Use FIR and PD for the baseline 16042021
            scalars = np.asarray([vals.FIR, vals.PD]).swapaxes(0, 1)
        elif no_input_channels == 3:
            scalars = np.asarray([vals.GWfr, vals.PD, vals.WP]).swapaxes(0, 1)
        assert scalars.shape == (conv_window_size, no_input_channels)
        X_scalars_single[j] = scalars
    X_scalars_test += [X_scalars_single]
    
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
    print('Predicting shot ' + str(shot))
    modelJoint.reset_states()
    array_sdir = model_dir + '/epoch_' + epoch_to_predict + '/network_np_out/' + 'test' + '/'
    if not os.path.isdir(array_sdir):
        os.makedirs(array_sdir)
    if with_elms:
        states, elms  = modelJoint.predict(np.asarray([X_scalars_test[0][:, :, :]]),batch_size=1,verbose=1)
    else:
        states = modelJoint.predict(np.asarray([X_scalars_test[0][:, :, :]]),batch_size=1,verbose=1)
    pred = states.squeeze()
    out_df = shot_df.copy()
    L_pred = repelem(pred[:,0], stride)
    D_pred = repelem(pred[:,1], stride)
    H_pred = repelem(pred[:,2], stride)

    if L_pred.shape[0] > intersect_times.shape[0]:
        remainder = L_pred.shape[0]%intersect_times.shape[0]
        L_pred = L_pred[:-remainder]    
        D_pred = D_pred[:-remainder]    
        H_pred = H_pred[:-remainder]

    out_df['LHD_det'] = np.zeros(shot_df.shape[0])
    out_df.loc[out_df['time'].round(5).isin(intersect_times), 'LHD_det'] = np.zeros(L_pred.shape[0])

    out_df['ELM_det'] = np.zeros(shot_df.shape[0])
    out_df.loc[out_df['time'].round(5).isin(intersect_times), 'ELM_det'] = np.zeros(L_pred.shape[0])

    out_df['ELM_prob'] = np.zeros(shot_df.shape[0])
    out_df.loc[out_df['time'].round(5).isin(intersect_times), 'ELM_prob'] = np.zeros(L_pred.shape[0])

    out_df['L_prob'] = np.zeros(shot_df.shape[0])
    out_df.loc[out_df['time'].round(5).isin(intersect_times), 'L_prob'] = L_pred

    out_df['D_prob'] = np.zeros(shot_df.shape[0])
    out_df.loc[out_df['time'].round(5).isin(intersect_times), 'D_prob'] = D_pred

    out_df['H_prob'] = np.zeros(shot_df.shape[0])
    out_df.loc[out_df['time'].round(5).isin(intersect_times), 'H_prob'] = H_pred

    out_df['SXR'] = np.zeros(shot_df.shape[0])    
 
    shots_dir = os.path.join(dirpath, 'data/Detected')
    if eval(git_test):
        out_df.to_csv(columns=['time', 'IP', 'SXR', 'FIR', 'PD', 'DML', 'LHD_det', 'ELM_det', 'L_prob', 'D_prob', 'H_prob', 'ELM_prob'], path_or_buf=os.path.join(shots_dir, machine_id + '_'  + str(shot) + '_ConvLSTM_det.csv'.format(train_machine, machine_id)), index=False)
    else:
        out_df.to_csv(columns=['time', 'IP', 'SXR', 'GWfr', 'PD', 'WP', 'LHD_det', 'ELM_det', 'L_prob', 'D_prob', 'H_prob', 'ELM_prob'], path_or_buf=os.path.join(shots_dir, machine_id + '_'  + str(shot) + '_{}2{}_ConvLSTM_det.csv'.format(train_machine, machine_id)), index=False)

def repelem(arr, num):
    arr = list(itertools.chain.from_iterable(itertools.repeat(x, num) for x in arr.tolist()))
    return np.asarray(arr)

if __name__ == '__main__':
    main()

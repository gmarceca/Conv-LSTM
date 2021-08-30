import sys
import os
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from helper_funcs import *
pd.options.mode.chained_assignment = None
# from keras.models import load_model
from matplotlib.backends.backend_pdf import PdfPages
#from plot_shot_results import plot_shot, plot_shot_full, plot_shot_cnn, plot_shot_simplified, plot_shot_lstm_sig_lab
#from plot_scores import plot_roc_curve, plot_kappa_histogram, out_sorted_scores
import csv
import datetime
import pickle

def main():
    exp_args = ['test']
    #exp_args = ['train', 'test']
    args = sys.argv
    model_dir = 'experiments/' + args[1]
    exp_n = args[1]
    epoch_to_predict = args[2]
    test_machine = args[3]
    exp_train_dic = load_dic(model_dir + '/params_data_train')
    exp_test_dic['machine_id'] = test_machine

    print('Setting for training: ', exp_train_dic)
    exp_test_dic = load_dic(model_dir + '/params_data_test')
    print('Setting for testing: ', exp_test_dic)

    c_offset = exp_train_dic['labelers']
    conv_window_size = exp_train_dic['conv_w_size']
    conv_w_offset = exp_train_dic['conv_w_offset']
    no_input_channels = exp_train_dic['no_input_channels']

    gaussian_time_window = 10e-4
    signal_sampling_rate = 1e4
    gaussian_hinterval = int(gaussian_time_window * signal_sampling_rate)
    print('Will count as correct ELM predictions within', gaussian_hinterval, 'time slices of ELM label')
    
    machine_id = exp_test_dic['machine_id']
   
    for exp_arg in exp_args:
        print('--------------------------------------------------------------------------------------------------' +
              str(exp_arg)+
              '--------------------------------------------------------------------------------------------------')

        exp_dic = load_dic(model_dir + '/params_data_' + exp_arg)
        labelers = exp_dic['labelers']
        c_offset = exp_dic['labelers']
        if machine_id == 'JET':
            exp_dic['shot_ids'] = [97927, 97803, 96532, 97828, 97830, 97832, 97835, 97971, 97465, 97466, 97468, 97469, 97473, 94785, 97476, 97477, 96713, 97745, 98005, 96993, 96745, 94315, 97396, 96885, 97398, 97399, 94968, 94969, 94971, 94973]
        elif machine_id == 'AUG'
            exp_dic['shot_ids'] = [35248, 35274, 35275, 35529, 35530, 35531, 35532, 35536, 35538, 35540, 35556, 35557,35564, 35582, 35584, 35604, 35607, 35837, 35852, 35967, 35972, 35975, 35537, 35539, 35561]
        shots = [str(s) for s in exp_dic['shot_ids']]

        conv_window_size = exp_dic['conv_w_size']
        conv_w_offset = exp_dic['conv_w_offset']
        no_input_channels = exp_dic['no_input_channels']
        pred_args = [model_dir, '/epoch_'+epoch_to_predict, exp_arg, labelers, conv_w_offset, shots, conv_window_size, no_input_channels, gaussian_hinterval, machine_id]
        predict(pred_args, exp_n, epoch_to_predict)
    
def predict(args, exp_n, epoch_to_predict):
    
    exp_arg = args[2]
    shots = args[5]
    labelers = args[3]
    gaussian_hinterval = args[8]
    data_dir = './labeled_data/'
    machine_id = args[9]
    data_dir = './labeled_data/' + machine_id + '/'
    fshots = {}
    conv_window_size = args[6]
    no_input_channels = args[7]
    conv_w_offset = int(args[4])
    intersect_times_d = {}
    model_dir = 'experiments/' + args[1]

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
        
        intersect_times = np.round(shot_df.time.values,5)
        if len(labelers) > 1:
            for k, labeler in enumerate(labelers):
                fshot_labeled = pd.read_csv(data_dir+ labeler +'/' + machine_id + '_'  + str(shot) + '_' + labeler + '_labeled.csv', encoding='utf-8')
                intersect_times = np.round(sorted(set(np.round(fshot_labeled.time.values,5)) & set(np.round(intersect_times,5))), 5)
        fshot_equalized = shot_df.loc[shot_df['time'].round(5).isin(intersect_times)]
        intersect_times = intersect_times[conv_window_size-conv_w_offset:len(intersect_times)-conv_w_offset]
        intersect_times_d[shot] = intersect_times
    
    model_dir = args[0]
    epoch_to_predict = args[1]
    
    print('Predicting on shot(s)...')
 
    pred_transitions =[]
    k_indexes =[]
    dice_cfs = []
    conf_mats = []
    conf_mets = []
    dice_cfs_dic = {}
    k_indexes_dic = {}
    pred_states = []
    pred_elms = []
    
    array_sdir = model_dir  + epoch_to_predict + '/network_np_out_'+machine_id+'_Xkhz/' + exp_arg + '/'

    for s_ind, s in enumerate(shots):
        pred_states += [np.load(array_sdir + str(s) + '_states_pred.npy')]
        #pred_elms += [np.load(array_sdir + str(s) + '_elms_pred.npy')]
        print(s, pred_states[-1].shape)
    thresholds = np.arange(105, step=10)/100
    collapsed_elms = []
    collapsed_elms_labels = []
    print('Post processing, saving .csv file(s)...')
    metrics_weights =[]
    dither_weights=[]
    concat_states_labels=[]
    concat_states_pred=[]
    # concat_tot_len_h = 0
    # concat_tot_len_d = 0
    # concat_tot_len_l = 0
    # concat_tot_len_noag = 0
    # concat_tot_len = 0
    ground_truth_concat = []
    consensus_concat = []
    states_pred_concat =[]
    labeler_elms_concat = []
    k_indexes = []
    concat_elms = []
    concat_elm_labels = []
    k_indexes_dic = {}
    stride = 10
    pdf_save_dir = model_dir + '/' + epoch_to_predict + '/plots/' + exp_arg + '/'
    myfile = open(model_dir + '/kappa_scores_exp_{}_epoch_{}_'+machine_id+'_Xkhz.txt'.format(exp_n, epoch_to_predict.split('_')[-1]), 'w')
    for i, shot in enumerate(shots):
        labeler_states = []
        labeler_elms = []
        print('----------------------------------------SHOT', str(shot), '-----------------------------------------')

        labeler = labelers[0]
        fshot = pd.read_csv(data_dir+ labeler +'/' + machine_id + '_'  + str(shot) + '_' + labeler + '_labeled.csv', encoding='utf-8')
        # fshot = remove_current_30kA(fshot)
        # fshot = remove_no_state(fshot)
        # fshot = remove_disruption_points(fshot)
        # fshot = fshot.reset_index(drop=True)
        intersect_times = intersect_times_d[shot]
        for k, labeler in enumerate(labelers):
            fshot_labeled = pd.read_csv(data_dir+ labeler +'/' + machine_id + '_'  + str(shot) + '_' + labeler + '_labeled.csv', encoding='utf-8')
            fshot_sliced = fshot_labeled.loc[fshot_labeled['time'].round(5).isin(intersect_times)]
            # print(len(fshot_sliced), len(fshot), len(intersect_times))
            labeler_states += [fshot_sliced['LHD_label'].values[0::stride]]
            #labeler_elms += [fshot_sliced['ELM_label'].values[0::stride]]
        # exit(0)
        fshot = fshot.loc[fshot['time'].round(5).isin(intersect_times)]
        
        # print(len(fshot), pred_states[0].shape)
        # exit(0)

        labeler_states = np.asarray(labeler_states)
        #labeler_elms = np.asarray(labeler_elms)
        pred_states_disc = np.argmax(pred_states[i][0,:], axis=1)
        pred_states_disc = pred_states_disc[:len(fshot_sliced)]
        pred_states_disc += 1 #necessary because argmax returns 0 to 2, while we want 1 to 3!
        # print(len(fshot), pred_states_disc.shape)
        # exit(0)
        #np.save('pred_score.npy', pred_states_disc) 
        #np.save('predictions_score/pred_{}_.npy'.format(shot), pred_states_disc)

        #pred_elms_single = pred_elms[i][0, :len(fshot_sliced),0]
        #concat_elms.extend(pred_elms_single)
        #concat_elm_labels.extend(fshot['ELM_label'].values)
        # print('here', pred_elms[i].shape, pred_elms_single.shape, len(concat_elms), len(concat_elm_labels))
       
        states_pred_concat.extend(pred_states_disc)
        assert(labeler_states.shape[1] == pred_states_disc.shape[0])
        
        ground_truth = calc_mode(labeler_states.swapaxes(0,1))
        # ground_truth_elms = calc_elm_mode(labeler_elms.swapaxes(0,1), gaussian_hinterval)
        
        ground_truth_concat.extend(ground_truth)
        #labeler_elms_mode = calc_elm_mode(labeler_elms.swapaxes(0,1), gaussian_hinterval)
        
        #labeler_elms_concat.extend(labeler_elms_mode)
        dice_cf = dice_coefficient(pred_states_disc, ground_truth)
        k_st = k_statistic(pred_states_disc, ground_truth)
        k_indexes += [k_st]
        print('dice_cf: ', dice_cf)
        print('kst', k_st)
        k_indexes_dic[shot] = k_st
        myfile.write("%s\n" % str(shot))
        myfile.write("%s\n" % np.array2string(k_st))
        consensus = calc_consensus(labeler_states.swapaxes(0,1)) #has -1 in locations which are not consensual, ie at least one person disagrees (case 3)
        consensus_concat.extend(consensus)
        
        # majority = calc_mode_remove_consensus(labeler_states.swapaxes(0,1)) #has -2 in locations of consensus, -1 in locations of total disagreement (case 2)
        # print('calculating with majority opinion removing consensus')
        # print(sum(majority == -1), sum(majority == -2), sum(majority > 0))
        
        # mode_labeler_states = ground_truth
        # mask1 = np.where(ground_truth != -1)[0]
        # temp2 = calc_elm_mode(labeler_elms.swapaxes(0,1), gaussian_hinterval)
        # # assert len(temp1) == len(temp2)
        # mask2 = np.where(temp2 != -1)[0]
        # mask = list(set(mask1) & set(mask2))
        # ground_truth = ground_truth[mask]
        # pred_states_disc = pred_states_disc[mask]
  
        #fshot['L_prob'] = pred_states[i][0,:, 0]
        #fshot['H_prob'] = pred_states[i][0,:, 2]
        #fshot['D_prob'] = pred_states[i][0,:, 1]
        #fshot['ELM_prob'] = pred_elms[i][0,: ,0]
        ## # fshot['ELM_det'] = pred_elms_disc
        ## fshot['ELM_label'] = labeler_elms_mode
        ## print(fshot, len(pred_states_disc))
        ## print(len(pred_states_disc), len(fshot), len(pred_states[i][0,:, 0]))
        #fshot['LHD_det'] = pred_states_disc
        #fshot['LHD_label'] = ground_truth
        ## 
        ## # concat_states_labels.extend(ground_truth)
        ## # concat_states_pred.extend(pred_states_disc)
        #
        #if not os.path.isdir(pdf_save_dir):
        #    os.makedirs(pdf_save_dir)
        #plot_fname = pdf_save_dir + 'shot_simp' + shot + '.png'
        #plot_shot_simplified(shot, fshot.copy(), plot_fname)
        ## plot_fname = pdf_save_dir + 'shot_full' + shot + '.pdf'
        ## plot_shot_full(shot, fshot.copy(), plot_fname, k_st)
        ## print('TCV_'  + str(shot) + '_CNN_det.csv')
        ## fshot_csv_fname = model_dir + '/' + epoch_to_predict + '/detector_csv_out/' + exp_arg + '/'
        ## if not os.path.isdir(fshot_csv_fname):
        ##     os.makedirs(fshot_csv_fname)
        ## 
        ## 
        #
        #
        ## pdf_save_dir 
        ## pdf_save_dir = model_dir + '/' + epoch_to_predict + '/' + exp_arg + '/' + 'shot_sig_lab' + shot + '.pdf'
        ## print(fshot.LHD_det.values[:300])
        ## plot_shot_lstm_sig_lab(shot, fshot.copy(), pdf_save_dir)
        #print('TCV_'  + str(shot) + '_LSTM_det.csv')
        #fshot_csv_fname = model_dir + '/' + epoch_to_predict + '/detector_csv_out/' + exp_arg + '/'
        #if not os.path.isdir(fshot_csv_fname):
        #    os.makedirs(fshot_csv_fname)
        #fshot.to_csv(columns=['time', 'IP', 'FIR', 'PD', 'DML', 'LHD_label', 'ELM_label', 'LHD_det', 'L_prob', 'D_prob', 'H_prob', 'ELM_prob'],
        #                  path_or_buf=fshot_csv_fname +  '/TCV_'  + str(shot) + '_LSTM_det.csv', index=False)
    
    k_indexes = np.asarray(k_indexes)
    
    ground_truth_concat = np.asarray(ground_truth_concat)
    consensus_concat = np.asarray(consensus_concat)
    states_pred_concat = np.asarray(states_pred_concat)
    #labeler_elms_concat = np.asarray(labeler_elms_concat)
    
    ground_truth_mask = np.where(ground_truth_concat!=-1)[0]
    #elm_label_mask = np.where(labeler_elms_concat!=-1)[0]
    #mask = list(set(ground_truth_mask) & set(elm_label_mask))

    ground_truth_concat = ground_truth_concat[ground_truth_mask]
    states_pred_concat = states_pred_concat[ground_truth_mask]
    consensus_concat = consensus_concat[ground_truth_mask] #should stay the same, as consensus is subset of ground truth
    
    avg_kappa = k_statistic(states_pred_concat, ground_truth_concat)
    avg_dice = dice_coefficient(states_pred_concat, ground_truth_concat)
    #with open(model_dir + '/kappa_scores_exp_{}_epoch_{}.txt'.format(exp_n, epoch_to_predict.split('_')[-1]), 'w') as f:
    #    f.write(np.array2string(avg_kappa))
    myfile.write("%s\n" % np.array2string(avg_kappa))
    myfile.close()
    print('Averaged dice predictions: ', avg_dice)
    print('Averaged kappa predictions: ', avg_kappa)
    print('Averaged kappa labelers: ', k_statistic(consensus_concat, ground_truth_concat))
    
    title = ''
    #concat_elms = np.asarray(concat_elms)
    #concat_elm_labels = np.asarray(concat_elm_labels)
    # roc_curve = get_roc_curve(concat_elms, concat_elm_labels, thresholds, gaussian_hinterval=10, signal_times=[])
    # roc_fname = pdf_save_dir + epoch_to_predict + exp_arg + 'roc_curve.pdf'
    # plot_roc_curve(roc_curve, thresholds, roc_fname, title)
    
    #histo_fname = pdf_save_dir + epoch_to_predict + exp_arg + 'k_ind_histogram.pdf'
    
    #plot_kappa_histogram(k_indexes, histo_fname, title)
    
    #fpath = model_dir + '/' + epoch_to_predict + '/' + exp_arg + 'k_ind_sorted_scores_'
    #out_sorted_scores(k_indexes_dic, fpath)
   
    
if __name__ == '__main__':
    main()

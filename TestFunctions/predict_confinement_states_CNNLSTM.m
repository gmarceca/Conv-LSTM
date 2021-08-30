function [data] = predict_confinement_states_CNNLSTM(shots)

addpath('../data_access');
data_dir = '../data/Detected/';

name_device = 'TCV';
for i=1:numel(shots)
    shot = shots(i);

    files_signals_search = fullfile('data','Detected',sprintf('%s_%d_signals.csv',name_device,shot));
    disp(files_signals_search);
    if ~exist(files_signals_search, 'file')
        % Get inputs
        data.shot = shot;
        data.machine = name_device;
        out = get_sig_data_TCV(shot,0);
        data.sigs = out;
        % Store inputs in a CSV table
        csv_handling('w','signals',data_dir, data, 1);
    end
    % Predict with CNNLSTM
    %% Detect PLasma Staes function
    command = sprintf('python algorithms/ConvLSTM/evaluate_model_from_detected_signals.py baseline_16042021_exp9 400 %d %s avg True', shot, name_device);
    system(command);
end

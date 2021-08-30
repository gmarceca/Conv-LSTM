import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import keras
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from custom_callbacks import ValidationKappaScore
from lstm_data_generator import *
from lstm_model import *
from helper_funcs import get_date_time_formatted
import random
import json
from collections import defaultdict

sys.stdout.flush()

dtime = get_date_time_formatted()
# id of experiment, could be an int or string
train_dir = './experiments/' + sys.argv[1]
if not os.path.isdir(train_dir):
    os.makedirs(train_dir)
print('Will save this model in', train_dir)

checkpoint_dir = train_dir +'/model_checkpoints/'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# TCV, JET or AUG. The first machine is the source domain while the rest are the target domains
machine_id = [sys.argv[2]]

# Get train and val shots split from json db
with open('train_and_val_shots.json', 'r') as a:
    shots_db = json.load(a)

train_shots = defaultdict(list)
val_shots = defaultdict(list)
try:
    for machine in machine_id:
        train_shots[machine] = shots_db[machine][0]['train_shots']
        val_shots[machine] = shots_db[machine][0]['val_shots']
except:
    print("machine_id {} is not in json database".format(machine_id))
    raise

# Check there are no training shots in val set
import collections
for im, machine in enumerate(machine_id):
    if im > 0: # If it is a target machine continue
        continue
    assert(len([item for item, count in collections.Counter(train_shots[machine]+val_shots[machine]).items() if count > 1])==0)

no_input_channels=3 # Using FIR, PD and DML
lstm_spread = 2000 # input seq length
num_classes = 2
timesteps = None
conv_w_size = 40
epoch_size = 64
bsize = 16*4
gaussian_hinterval = 5 # time steps window for labelling smoothing
no_epochs = 100
stride = 10
conv_w_offset = 20
normalization = 'z_GCS'
data_augm = False
arch_with_elms = False # Initialize model arc with or without ELMs nodes
labelers = sys.argv[3].split(',') # Labeler name: TCV_64647_XXX_labeled.csv: e.g XXX = apau_and_marceca, XXX = detrend, etc
shuffle=True

# -------- Create and compile model --------
convlstm = ConvLSTM(bsize, conv_w_size, no_input_channels, timesteps, arch_with_elms)
modelJoint = convlstm.create_architecture()
modelJoint.compile(loss={'out_states':'categorical_crossentropy'}, optimizer='adam',metrics={'out_states':'categorical_accuracy'})
modelJoint.summary()

# For a quick test:
#train_shots['TCV'] = [57000]
#train_shots['AUG'] = [35538]
#val_shots['TCV'] = [57000]
#val_shots['AUG'] = [35538]

for machine in machine_id:
    print('randomized train shot ids for machine {} '.format(machine), train_shots[machine], len(train_shots[machine]))
    print('randomized val shots ids for machine {} '.format(machine), val_shots[machine], len(val_shots[machine]))
    
params_lstm_random = {
            'batch_size': int(bsize),
            'lstm_time_spread': int(lstm_spread),
            'epoch_size': epoch_size,
            'no_input_channels' : no_input_channels,
            'conv_w_size':conv_w_size,
            'gaussian_hinterval': gaussian_hinterval,
            'no_classes': num_classes,
            'stride':int(stride),
            'labelers':labelers,
            'shuffle':shuffle,
            'conv_w_offset':conv_w_offset,
            'machine_id':machine_id,
            'normalization':normalization,
            'data_augm': data_augm}
print('experiment parameters', params_lstm_random)
params_random_train = {'shot_ids': train_shots}
params_random_train.update(params_lstm_random)
params_random_val = {'shot_ids': val_shots}
params_random_val.update(params_lstm_random)

print('preparing training_generator')
training_generator = LSTMDataGenerator(**params_random_train)
gen_train = next(iter(training_generator))

print('preparing val_generator')
val_generator = LSTMDataGenerator(**params_random_val)
gen_val = next(iter(val_generator))

# Add properties (not used on data generator)
params_random_train = {'with_elms': arch_with_elms}
params_random_train.update(params_lstm_random)

save_dic(params_random_train, train_dir + '/params_data_train')
save_dic(params_random_val, train_dir + '/params_data_test')

saveCheckpoint = keras.callbacks.ModelCheckpoint(filepath= checkpoint_dir + 'weights.{epoch:02d}.h5', period=2)
machines = ['TCV', 'AUG', 'JET']
kappascore = ValidationKappaScore(sys.argv[1], machines, normalization, arch_with_elms)

modelJoint.fit_generator(generator = gen_train, steps_per_epoch=epoch_size, epochs=no_epochs, validation_data=gen_val, validation_steps=bsize, callbacks=[saveCheckpoint, kappascore]) #,tb ,validation_data=gen_val, validation_steps=bsize

modelJoint.save_weights(checkpoint_dir + 'weights.' + str(no_epochs) + ".h5")
sys.stdout.flush()

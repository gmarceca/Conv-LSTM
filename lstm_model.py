import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Layer
from tensorflow.keras.layers import Input, MaxPooling1D, Flatten, concatenate, Dense, Dropout, BatchNormalization, Conv1D, Activation, Lambda
from tensorflow.keras.models import Model

TIMESTEPS = 2000
INPUT_CHANNELS = 3
CONV_WINDOW_SIZE = 40
BATCH_SIZE = 64
WITH_ELMS=False

class ConvLSTM:
    def __init__(self, bsize=BATCH_SIZE, conv_w_size=CONV_WINDOW_SIZE, no_input_channels=INPUT_CHANNELS, timesteps=TIMESTEPS, with_elms=WITH_ELMS):
        self.bsize = bsize
        self.conv_w_size = conv_w_size
        self.no_input_channels = no_input_channels
        self.timesteps = timesteps
        self.with_elms = with_elms 
        self.model = None

    def create_architecture(self):
    
        conv_input = Input(shape=(self.conv_w_size, self.no_input_channels,), dtype='float32', name='conv_input')
        
        x_conv = Conv1D(32, 3, activation='relu', padding='same')(conv_input)
        x_conv = Conv1D(64, 3, activation='relu', padding='same')(x_conv)
        x_conv = Dropout(.5)(x_conv)
        x_conv = MaxPooling1D(2)(x_conv)
        
        conv_out = Flatten()(x_conv)
        conv_out = Dense(16, activation='relu')(conv_out)
        modelCNN = Model(inputs=[conv_input], outputs= [conv_out])
        
        # Prepare input sequence in a sliding window settings. Each timestep consists in a window of conv_w_size steps.
        joint_input = Input(batch_shape=(int(self.bsize),self.timesteps,self.conv_w_size,self.no_input_channels), dtype='float32', name='input_signals')

        # Feature extractor
        modelJoined = TimeDistributed(modelCNN)(joint_input)
        main_output = LSTM(32, return_sequences=True, stateful=False)(modelJoined)
        main_output = TimeDistributed(Dense(8, activation='relu'))(main_output)
        main_output = Dropout(.5)(main_output)
        # State classifier
        modelJoinedStates = TimeDistributed(Dense(3, activation='softmax'),name='out_states')(main_output)
        
        if self.with_elms: # Baseline 16042021 model used this settings
            modelJoinedElms = TimeDistributed(Dense(2, activation='softmax'), name='out_elms')(main_output)
            self.model = Model(inputs=[joint_input], outputs= [modelJoinedStates,modelJoinedElms])
        else:
            self.model = Model(inputs=[joint_input], outputs= [modelJoinedStates])

        return self.model

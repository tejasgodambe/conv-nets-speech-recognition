#!/usr/bin/python3

# Author: Tejas Godambe 
# Date: April 2017

import theano 
import theano.tensor as T 
import sys, os, time
import numpy as np
import keras
from subprocess import Popen, PIPE
from dataGenerator_CNN import dataGenerator_CNN
from dataGenerator_CNN import dataGenerator_CNN
#from custom_crossentropy import custom_crossentropy
import kaldiIO 
from saveModel import saveModel
from average_DNN_predictions import geometric_mean, arithmetic_mean

############# USER CONFIGURABLE PARAMS STARTS #############
## Learning parameters
learning = {'rate' : 0.08,
            'batchSize' : 256}

## Input directories
data_tr = 'train_tr95'
data_cv = 'train_cv05'
sgmm='/home/pavan/2016h2/en-US-wc-sou-dd2/exp/sgmm2_4a_1500'
ali_tr='/home/pavan/2016h2/en-US-wc-sou-dd2/exp/sgmm2_4a_1500_ali_tr95'
ali_cv='/home/pavan/2016h2/en-US-wc-sou-dd2/exp/sgmm2_4a_1500_ali_cv05'

## Output directories
outdir = 'outdir_en-US_CNN'
model_out = outdir + '/' + 'dnn.nnet'
model_out_h5 = model_out + '.h5'
############# USER CONFIGURABLE PARAMS ENDS #############


## Main code
if __name__ == '__main__': 
    os.makedirs (outdir, exist_ok=True)
    
    ## Initialize data generator
    trGen = dataGenerator_CNN (data_cv, ali_cv, sgmm, learning['batchSize'])
    cvGen = dataGenerator_CNN (data_cv, ali_cv, sgmm, learning['batchSize'])
    
    ## Define DNN architecture and initialize weights
    m = keras.models.Sequential([
                    keras.layers.convolutional.Conv2D(64, 3, 3, subsample=(1,1), 
                        border_mode='same', activation='relu', input_shape=(1,39,11)),
                    keras.layers.core.Flatten(),
                    keras.layers.Dense(256, activation='relu'),
                    keras.layers.Dense(1375, activation='softmax')])
    
    print (m.summary())
    
    ## Configure DNN training
    s = keras.optimizers.SGD(lr=learning['rate'], decay=0, momentum=0.5, nesterov=True)
    m.compile(loss='categorical_crossentropy', optimizer=s, metrics=['accuracy'])
    
    print ('Learning rate: %f' % learning['rate'])
    
    ## Keras callbacks
    filepath = outdir + '/weights.{epoch:02d}-{val_acc:.2f}.h5'
    # to save weights after each epoch
    checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', 
                   verbose=1, save_best_only=False, mode='auto')
    # to reduce learning rate if loss marginally decreases for consecutive epochs 
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, 
                patience=1, epsilon=0.25, mode='auto', verbose=1, min_lr=0)
    # to stop training before overfitting happens
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', 
                     min_delta=0.005, patience=3, verbose=1, mode='auto')
    
    ## DNN training 
    m.fit_generator (trGen, samples_per_epoch=trGen.numFeats, 
          max_q_size=1000, nb_worker=10, pickle_safe=False,
          validation_data=cvGen, nb_val_samples=cvGen.numFeats,
          nb_epoch=30, verbose=1, callbacks=[checkpointer, reduce_lr, early_stopping]) 
    
    ## Save final weights
    m.save_weights (model_out_h5, overwrite=True)
    saveModel (m, model_out) # convert weights in h5 to txt format
    
    print ('Learning finished ...')

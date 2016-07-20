# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import process_data as prepare
import numpy as np
from keras.callbacks import ModelCheckpoint
import pickle
import sys
sys.setrecursionlimit(10000) # to be able to pickle Theano compiled functions
seed = 113

def create_model():
    model = Sequential()
    model.add(LSTM(input_dim=13, output_dim=128, return_sequences=True))
    model.add(LSTM(input_dim=128, output_dim=128, return_sequences=True))
    model.add(Dropout(0.05))
    model.add(LSTM(input_dim=128, output_dim=128, return_sequences=True))
    model.add(Dropout(0.05))
    model.add(LSTM(input_dim=128, output_dim=128, return_sequences=False))
    model.add(Dense(output_dim=1,init='uniform'))
    model.add(Activation("sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
    return model

TEST = 1 # 1 for evaluation only, 0 for training

mean_std = np.load('mat/mean_std.npz')
(mean,std) = (mean_std['mean'],mean_std['std'])

if TEST == 0:
    print "load train data..."
    npzfile_train = np.load('mat/train_data_all.npz')
    (X_train,Y_train)= (npzfile_train['data'],npzfile_train['label'])
    for i in range(X_train.shape[0]):
        X_train[i,:,:]  = X_train[i,:,:] - mean / std
    print " train data"
    print X_train.shape
    print Y_train.shape
    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(Y_train)

npzfile_test = np.load('mat/test_data_all.npz')
(X_test,Y_test) = (npzfile_test['data'],npzfile_test['label'])
print " test data"
print X_test.shape
print Y_test.shape
np.random.seed(seed)
np.random.shuffle(X_test)
np.random.seed(seed)
np.random.shuffle(Y_test)
for i in range(X_test.shape[0]):
    X_test[i,:,:]  = X_test[i,:,:] - mean / std

print "create model..."


model = create_model()
if TEST:
    print "load test"
    model.load_weights('temp/_model_weights99.hdf5')



if TEST == 0:
    print 'training'
    checkpointer = ModelCheckpoint(filepath="temp/_model_weights{epoch:02d}.hdf5", verbose=1, save_best_only=False)
    model.fit(X_train, Y_train, batch_size=128, nb_epoch=100,show_accuracy=True, verbose=2,callbacks=[checkpointer])
print 'testing'
scores = model.evaluate(X_test, Y_test,verbose=1,batch_size=128)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

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
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", class_mode="binary")
    return model



'''
model_loaded = create_model()
model_loaded.load_weights('temp/model_weights360.hdf5')
'''
TEST = 1
print "load data..."
npzfile = np.load('temp/train_data.npz')
print "split data..."
(X_train,Y_train,X_test,Y_test) = prepare.train_test_split(npzfile)
print " train data"
print X_train.shape
print Y_train.shape
print " test data"
print X_test.shape
print X_test.shape
print "create model..."
model = create_model()
pickle.dump(model, open('/tmp/model.pkl', 'wb'))
if TEST:
    print "load test"
    model.load_weights('temp/model_weights324.hdf5')
seed = 113
np.random.seed(seed)
np.random.shuffle(X_train)
np.random.seed(seed)
np.random.shuffle(Y_train)
np.random.seed(seed)
np.random.shuffle(X_test)
np.random.seed(seed)
np.random.shuffle(Y_test)
if TEST == 0:
    checkpointer = ModelCheckpoint(filepath="temp/model_weights{epoch:02d}.hdf5", verbose=1, save_best_only=False)
    model.fit(X_train, Y_train, batch_size=128, nb_epoch=2000,show_accuracy=True, verbose=2,callbacks=[checkpointer])
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
#model.save_weights('/tmp/model_weights.hdf5')
print('Test score:', score[0])
print('Test accuracy:', score[1])

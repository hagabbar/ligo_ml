#Deep Learning Regression algorithm to predict scattering noise in aLIGO Detectors
#Author: Hunter Gabbard
#Max Planck Institute for Gravitational Physics
# %run neural_net.py -d data_Dec2-Dec3.hdf

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
import sys
from math import exp, log
import tensorflow as tf
from keras.callbacks import EarlyStopping
import h5py
from numpy import *
from matplotlib import use
use('Agg')
from matplotlib import pyplot as pl
import argparse

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


def keras_nn(X_train, X_test, Y_train, Y_test):

    #Compiling Neural Network
    model = Sequential()
    model.add(Dense(81, input_dim=X_data.shape[1]))
    model.add(Activation('relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')

    #Original model
    #model = Sequential()
    #model.add(Dense(27, input_dim=X_data.shape[1]))
    #model.add(Activation('softsign'))
    #model.add(Dense(300, activation='softsign'))
    #model.add(Dense(500, activation='softsign'))
    #model.add(Dense(700, activation='softsign'))
    #model.add(Dense(500, activation='softsign'))
    #model.add(Dense(200, activation='softsign'))
    #model.add(Dense(1))
    #model.compile(loss='mean_squared_error', optimizer='rmsprop')

    #Fitting model to training set
    model.fit(X_train, Y_train, nb_epoch=50, batch_size=32)
    
    #Saving model for later use
    model.save_weights('model.h5')
   
    #Predicitng on test set
    score = model.predict(X_test, batch_size=32)
    return score

def plotter(predic, Y_test, f_title):
    #Comparing RMS pred vs. True RMS
    pl.hist(predic, bins=100, alpha=0.6, label='predicted')
    pl.hist(Y_test, bins=100, alpha=0.6, label='True')
    pl.title('Predicted and True')
    pl.legend()
    pl.savefig('/home/hunter.gabbard/public_html/Detchar/scattering/normalized/comb_hist_%s.png' % f_title)
    pl.close()

    #Plotting the accuracy
    pl.hist((predic - Y_test), bins=100)
    pl.title('Accuracy')
    pl.savefig('/home/hunter.gabbard/public_html/Detchar/scattering/normalized/accuracy_hist_%s.png' % f_title)
    pl.close()

    #Scater Plot of Actual as a function of Predicted
    pl.scatter(predic,Y_test)
    pl.xlim(0,1)
    pl.ylim(0,1)
    pl.plot(linspace(0,1,100), linspace(0,1,100))
    pl.title('Actual vs. Predicted')
    pl.xlabel('Predicted')
    pl.ylabel('Actual')
    pl.savefig('/home/hunter.gabbard/public_html/Detchar/scattering/normalized/comb_scat_%s.png' % f_title)
    pl.close()

    #RMS plot
    pl.plot(predic, alpha=0.6, label='predicted')
    pl.plot(Y_test, alpha=0.6, label='True')
    pl.legend()
    pl.savefig('/home/hunter.gabbard/public_html/Detchar/scattering/normalized/RMS_timeseries_test.png')
    pl.close()


if __name__ == '__main__':
    #construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
            help="path to input dataset")
    args = vars(ap.parse_args())

    #Initializing parameters
    f_title = 'test'
    t_perc = 0.55
    d = h5py.File(args['dataset'], 'r')
    X_data = d['X_data']
    Y_data = d['Y_data']
    Y_data = log(Y_data)
    Y_data = (Y_data - Y_data.min()) / (Y_data.max() - Y_data.min())   

    #Defining training and testing sets
    X_train, X_test = X_data[:int(len(X_data)*t_perc),:]/X_data[:int(len(X_data)*t_perc),:].max(), X_data[int(len(X_data)*t_perc):len(X_data),:]/X_data[int(len(X_data)*t_perc):len(X_data),:].max()
    Y_train, Y_test = Y_data[:int(len(X_data)*t_perc),:], Y_data[int(len(X_data)*t_perc):len(X_data),:]

    #Making predicted DARM RMS time series
    predic = keras_nn(X_train, X_test, Y_train, Y_test)

    #Plotting the results and other figures of merit 
    plotter(predic, Y_test, f_title)

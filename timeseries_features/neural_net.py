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
from keras.models import load_model
import sys
import os
from math import exp, log
import tensorflow as tf
from keras.callbacks import EarlyStopping
import h5py
import numpy as np
from numpy import *
from matplotlib import use
use('Agg')
from matplotlib import pyplot as pl
import argparse
import datetime

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


def keras_nn(X_train, X_test, Y_train, Y_test):

    #Compiling Neural Network
    model = Sequential()
    model.add(Dense(1500, input_dim=X_train.shape[1]))
    model.add(Activation('relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])

    #Original model
    #model = Sequential()
    #model.add(Dense(500, input_dim=X_train.shape[1]))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(300, activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(300, activation='relu'))
    #model.add(Dense(300, activation='relu'))
    #model.add(Dense(1))
    #model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

    #Fitting model to training set
    hist = model.fit(X_train, Y_train, nb_epoch=1000, batch_size=1500, shuffle=True, show_accuracy=True)
    print(hist.history)    
   
    #Predicitng on test set
    print("[INFO] Predicting on test set...")
    score = model.predict(X_test, batch_size=32)
   
    #Printing summary of model parameters
    model.summary()

    #Calculate the overall score of the run
    perf_score = np.mean(Y_test * abs(score.flatten() - Y_test))
    print('Your overall score for the run is (the lower, the better): %f' % perf_score)
   
    return score, perf_score, model, hist.history

def plotter(predic, Y_test, out_dir, now, model, hist):

    #Saving model to hdf file for later use
    model.save('%s/run_%s/nn_model.hdf' % (out_dir,now))
    np.save('%s/run_%s/hist.npy' % (out_dir,now), hist)

    #Comparing RMS pred vs. True RMS
    pl.hist(predic, bins=100, alpha=0.6, label='predicted')
    pl.hist(Y_test, bins=100, alpha=0.6, label='True')
    pl.title('Predicted and True')
    pl.legend()
    pl.savefig('%s/run_%s/comb_hist.png' % (out_dir,now))
    pl.close()

    #RMS plot
    pl.plot(predic[:300], alpha=0.6, label='predicted')
    pl.plot(Y_test[:300], alpha=0.6, label='True')
    pl.legend()
    pl.savefig('%s/run_%s/RMS_timeseries_600s.png' % (out_dir,now))
    pl.close()

    #Plotting the accuracy
    pl.hist((predic.flatten() - Y_test), bins=100)
    pl.title('Accuracy')
    pl.savefig('%s/run_%s/accuracy_hist.png' % (out_dir,now))
    pl.close()

    #Scater Plot of Actual as a function of Predicted
    pl.scatter(predic,Y_test)
    pl.xlim(0,1)
    pl.ylim(0,1)
    pl.plot(linspace(0,1,100), linspace(0,1,100))
    pl.title('Actual vs. Predicted')
    pl.xlabel('Predicted')
    pl.ylabel('Actual')
    pl.savefig('%s/run_%s/comb_scat.png' % (out_dir,now))
    pl.close()

def load_data(X_data,Y_data,t_perc):
    #Creating a mask
    Y_data = Y_data
    mask = Y_data<1.2
    Y_data = Y_data[mask]
    X_data = X_data[:,mask]

    #Initializing parameters
    X_data = X_data.T[:,246:281]       #[:,246:269]
    Y_data = Y_data.T

    Y_data = log(Y_data)
    Y_data = (Y_data - Y_data.min()) / (Y_data.max() - Y_data.min())

    #Defining training and testing sets
    X_train, X_test = X_data[:int(len(X_data)*t_perc),:]/X_data[:int(len(X_data)*t_perc),:].max(), X_data[int(len(X_data)*t_perc):len(X_data),:]/X_data[int(len(X_data)*t_perc):len(X_data),:].max()
    Y_train, Y_test = Y_data[:int(len(X_data)*t_perc)], Y_data[int(len(X_data)*t_perc):len(X_data)]

    return X_train, X_test, Y_train, Y_test

def time_stat(Y_test):
    for idx in range(-18,18):
        Y_predicted = Y_test[20+idx:-20+idx]
        Y_true = Y_test[20:-20]
        result = np.mean(Y_true * abs(Y_predicted - Y_true))
        #print result, idx
    return result

def shuffle_stat(Y_test):
    Y_predicted = Y_test
    Y_predicted = np.random.choice(Y_predicted,size=len(Y_test),replace=True)[:10000]
    Y_true = Y_test[:10000]
    result = np.mean(Y_true * abs(Y_predicted - Y_true))
    print("Shuffle stat result: %f" % result)
    return result

def old_nn(X_train, X_test, Y_train, Y_test, s_mod):
    #Loading saved nn model
    model = load_model('%s' % s_mod)
    
    #Predicitng on test set
    print("[INFO] Predicting on test set...")
    score = model.predict(X_test, batch_size=32)

    #Printing summary of model parameters
    model.summary()

    #Calculate the overall score of the run
    perf_score = np.mean(Y_test * abs(score.flatten() - Y_test))
    print('Your overall score for the run is (the lower, the better): %f' % perf_score)

    return score, perf_score, model

if __name__ == '__main__':
    #construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
            help="path to input data. (e.g. data1.hdf,data2.hdf,...")
    #ap.add_argument("-da", "--aux_dataset", required=True,
    #        help="path to aux input dataset")
    #ap.add_argument("-dd", "--darm_dataset", required=True,
    #        help="path to darm input dataset")
    ap.add_argument("-c", "--channel_list", required=True,
            help="List of channels used to generate dataset")
    ap.add_argument("-o", "--output_directory", required=True,
            help="output directory")
    ap.add_argument("-t", "--test_only", required=True,
            help="Would you like to use a previous model and test on that (True or False)?")
    ap.add_argument("-m", "--path_to_model", required=False,
            help="path to old nn model")
    ap.add_argument("-p", "--tt_split", required=True,
            help="test/train split percentage")
    args = vars(ap.parse_args())


    #Specify parameters
    train_perc =float(args['tt_split'])
    d_trig = np.load(args['dataset'].split(',')[0])
    d_darm = np.load(args['dataset'].split(',')[1])
    
    #d_trig = []
    #d_trig = np.asarray(d_trig)
    #d_darm = []
    #d_darm = np.asarray(d_darm)
    #for idx, fi in enumerate(args['aux_dataset'].split(',')):
    #    if idx == 0:
    #        d_trig = np.load(fi)
    #    elif idx > 0:
    #        d_trig = np.hstack((d_trig,np.load(fi)))
    #for idx, fi in enumerate(args['darm_dataset'].split(',')):
    #    if idx == 0:
    #        d_darm = np.load(fi)[:len(np.load(fi))-1]
    #    elif idx > 0:
    #        d_darm = np.hstack((d_darm,np.load(fi)[:len(np.load(fi))-1]))

    out_dir = args['output_directory']
    #Get current time for time stamp labels
    now = datetime.datetime.now() 

    #Load data and seperate into training/test sets
    X_train, X_test, Y_train, Y_test = load_data(d_trig, d_darm, train_perc)

    #Making predicted DARM RMS time series and retrieving overall score of run
    if args['test_only'] == 'True':
        predic, perf_score, mod = old_nn(X_train, X_test, Y_train, Y_test, args['path_to_model'])
        
    elif args['test_only'] == 'False':
        predic, perf_score, mod, hist = keras_nn(X_train, X_test, Y_train, Y_test)

    #Make output directory
    os.makedirs('%s/run_%s' % (out_dir,now))

    #Plotting the results and other figures of merit 
    if args['test_only'] == 'True':
        hist = []
        plotter(predic, Y_test, out_dir, now, mod, hist)
    elif args['test_only'] == 'False':
        plotter(predic, Y_test, out_dir, now, mod, hist)

    #Calculating the time statistic
    time_stat(Y_test)
    
    #Save the overall score of the run
    np.save('%s/run_%s/run_score.npy' % (out_dir,now), perf_score)

    #Calculating the shuffle statistic
    shuffle_stat(Y_test)    

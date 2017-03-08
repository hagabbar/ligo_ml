#Deep Learning Regression algorithm to predict scattering noise in aLIGO Detectors
#Author: Hunter Gabbard
#Max Planck Institute for Gravitational Physics
# %run neural_net.py -d data_Dec2-Dec3.hdf

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, GaussianDropout
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
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


def keras_nn(learning_rate, epochs, b_size, X_train, X_test, Y_train, Y_test):
    print "Training..."
    drop_rate = 0.2
    ret_rate = 1 - drop_rate
    act='relu'
    #Compiling Neural Network
    model = Sequential()
    model.add(Dense(int(X_train.shape[1]/ret_rate), input_dim=X_train.shape[1]))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(GaussianDropout(drop_rate))

    model.add(Dense(int(X_train.shape[1]/ret_rate)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(GaussianDropout(drop_rate))

    model.add(Dense(int(X_train.shape[1]/ret_rate)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(GaussianDropout(drop_rate))

    model.add(Dense(int(X_train.shape[1]/ret_rate)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(GaussianDropout(drop_rate))

    model.add(Dense(int(X_train.shape[1]/ret_rate)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(GaussianDropout(drop_rate))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])


    #Fitting model to training set
    hist = model.fit(X_train, Y_train, nb_epoch=epochs, batch_size=b_size, shuffle=True, show_accuracy=True)
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

def plotter(run_num, predic, Y_test, out_dir, now, model, hist):

    #Saving model to hdf file for later use
    model.save('%s/run_%s/nn_model.hdf' % (out_dir,now))
    np.save('%s/run_%s/hist.npy' % (out_dir,now), hist)

    print 'plotting training metrics'
    print hist.keys()
    for i, metric in enumerate(['loss', 'acc']):
        mname = metric.replace('acc', 'accuracy')
        pl.figure(run_num+i)
        pl.plot(hist[metric], label='Training', alpha=0.4)
        #pl.plot(hist['val_'+metric], label='Validation', alpha=0.4)
        pl.legend(frameon=True, loc='center right')
        pl.xlabel('Epoch')
        pl.ylabel(mname.replace('_', ' '))
        pl.ylim(ymin=0.)
        pl.savefig('%s/run_%s/%s_vs_epoch.png' % (out_dir, now, mname[0:4]))
        pl.close()

    #Comparing RMS pred vs. True RMS
    pl.figure(run_num)
    pl.hist(predic, bins=100, alpha=0.4, label='predicted')
    pl.hist(Y_test, bins=100, alpha=0.4, label='True')
    pl.title('Predicted and True')
    pl.legend()
    pl.savefig('%s/run_%s/comb_hist.png' % (out_dir,now))
    pl.close()

    #RMS plot
    pl.figure(run_num)
    pl.plot(predic[:300], alpha=0.4, label='predicted')
    pl.plot(Y_test[:300], alpha=0.4, label='True')
    pl.legend()
    pl.savefig('%s/run_%s/RMS_timeseries_600s.png' % (out_dir,now))
    pl.close()

    #Plotting the accuracy
    pl.figure(run_num)
    pl.hist((predic.flatten() - Y_test), bins=100, alpha=0.5)
    pl.title('Accuracy')
    pl.savefig('%s/run_%s/accuracy_hist.png' % (out_dir,now))
    pl.close()

    #Scater Plot of Actual as a function of Predicted
    pl.figure(run_num)
    pl.scatter(predic,Y_test, s=10, alpha=0.4)
    pl.xlim(0,1)
    pl.ylim(0,1)
    pl.plot(linspace(0,1,100), linspace(0,1,100))
    pl.title('Actual vs. Predicted')
    pl.xlabel('Predicted')
    pl.ylabel('Actual')
    pl.savefig('%s/run_%s/comb_scat.png' % (out_dir,now))
    pl.close()

def load_data(d_trig,d_darm,t_perc):
    #Combining aux channel data into one array and loading darm data into one array
    Y_data = np.load(d_darm)
    for idx,fi in enumerate(d_trig):
        if idx == 0:
            X_data = np.load(d_trig[idx])
        elif idx > 0:
            X_data = np.vstack((X_data,np.load(d_trig[idx])))

    #Creating a mask
    Y_data = Y_data
    mask = Y_data < 8      #<1.2
    Y_data = Y_data[mask]
    X_data = X_data[:,mask]

    #Initializing parameters
    X_data = X_data.T       #[:,246:269]
    Y_data = Y_data.T
    print Y_data.shape
     
    Y_transform = Y_data

    Y_data = log(Y_data)
    Y_data = (Y_data - Y_data.min()) / (Y_data.max() - Y_data.min())

    #Defining training and testing sets
    X_train, X_test = X_data[:int(len(X_data)*t_perc),:]/X_data[:int(len(X_data)*t_perc),:].max(), X_data[int(len(X_data)*t_perc):len(X_data),:]/X_data[int(len(X_data)*t_perc):len(X_data),:].max()
    Y_train, Y_test = Y_data[:int(len(X_data)*t_perc)], Y_data[int(len(X_data)*t_perc):len(X_data)]

    return X_train, X_test, Y_train, Y_test, Y_transform

def load_data_testing(d_trig, d_darm, Y_transform):
    #Combining aux channel data into one array and loading darm data into one array
    Y_data = np.load(d_darm)
    for idx,fi in enumerate(d_trig):
        if idx == 0:
            X_data = np.load(d_trig[idx])
        elif idx > 0:
            X_data = np.vstack((X_data,np.load(d_trig[idx])))

    #Initializing parameters
    X_data = X_data.T
    Y_data = np.load(Y_data).T

    Y_data = log(Y_data)

    Y_transform = log(Y_transform)
    Y_transform = (Y_data - Y_transform.min()) / (Y_transform.max() - Y_transform.min())

    Y_test = (Y_data - Y_data.min()) / (Y_data.max() - Y_data.min())
    X_data = X_data/X_data.max()
    

    X_test = X_data

    return X_test, Y_test, Y_transform 

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

def old_nn(learning_rate, epochs, X_test, Y_test, s_mod):
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
    #Get current time for time stamp labels
    cur_time = datetime.datetime.now()

    #construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-dd", "--darm-dataset", required=True, nargs='+',
            help="path to darms data. (e.g. data1.npy,data2.npy,...")
    ap.add_argument("-da", "--aux-dataset", required=True, nargs='+',
            help="path to auxiliary channel data. (e.g. data1.npy,data2.npy,...")
    ap.add_argument("-c", "--channel_list", required=True,
            help="List of channels used to generate dataset")
    ap.add_argument("-o", "--output_directory", required=True,
            help="output directory")
    ap.add_argument("-t", "--test_only", required=True, type=str,
            help="Would you like to use a previous model and test on that (True or False)?")
    ap.add_argument("-m", "--path_to_model", required=False,
            help="path to old nn model")
    ap.add_argument("-od", "--orig_darm", required=False,
            help="path to trained model darm data")
    ap.add_argument("-p", "--tt_split", required=True, type=float,
            help="test/train split percentage")
    ap.add_argument("-e", "--epochs", required=False, default=100, type=int,
            help="Number of epochs for training")
    ap.add_argument("-bs", "--batch-size", required=False, default=32, type=int,
            help="Batch size for the training process (number of samples to use in each gradient descent step). Default 32")
    ap.add_argument("-r", "--run-number", required=False, default=0, type=int,
            help="If performing multiple runs on same machine, specify a unique number for each run (must be greater than zero)")
    ap.add_argument("--learning-rate", type=float, default=0.01,
        help="Learning rate. Default 0.01")
    ap.add_argument("--dropout-fraction", type=float, default=0.,
        help="Amount of Gaussian dropout noise to use in training. Default 0 (no noise)")
    ap.add_argument("-u", "--usertag", required=False, default=cur_time, type=str,
            help="label for given run")
    args = ap.parse_args()


    #Specify parameters
    train_perc = args.tt_split
    d_trig = args.aux_dataset    
    epochs = args.epochs
    b_size = args.batch_size
    d_darm = args.darm_dataset[0]
    if args.test_only == 'True':
        Y_transform = np.load(args.orig_darm)
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

    out_dir = args.output_directory
    run_num = args.run_number
    #Get user tag for run
    now = args.usertag 

    #Load data and seperate into training/test sets
    if args.test_only == 'True':
        X_test, Y_test, Y_transform = load_data_testing(d_trig, d_darm, Y_transform)

    elif args.test_only == 'False':
        X_train, X_test, Y_train, Y_test, Y_transform = load_data(d_trig, d_darm, train_perc)

    #Making predicted DARM RMS time series and retrieving overall score of run
    if args.test_only == 'True':
        predic, perf_score, mod = old_nn(args.learning_rate, epochs, X_test, Y_transform, args.path_to_model)
        
    elif args.test_only == 'False':
        predic, perf_score, mod, hist = keras_nn(args.learning_rate, epochs, b_size, X_train, X_test, Y_train, Y_test)

    #Make output directory
    os.makedirs('%s/run_%s' % (out_dir,now))

    #Plotting the results and other figures of merit 
    if args.test_only == 'True':
        hist = []
        plotter(run_num, predic, Y_transform, out_dir, now, mod, hist)
    elif args.test_only == 'False':
        plotter(run_num, predic, Y_test, out_dir, now, mod, hist)

    #Calculating the time statistic
    time_stat(Y_test)
    
    #Save the overall score of the run
    np.save('%s/run_%s/run_score.npy' % (out_dir,now), predic)

    if args.test_only == 'False':
        #Save the original Y_data of this run so that things scale well with later blind testing
        np.save('%s/run_%s/darm_data.npy' % (out_dir,now), Y_transform)

    #Calculating the shuffle statistic
    shuffle_stat(Y_test)    

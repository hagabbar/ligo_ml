#Support Vector Machine to do regression on DARM rms
#Author: Hunter Gabbard
#Max Planck Institute for Gravitational Physics
from sklearn.svm import SVR
import numpy as np
import sys
import tensorflow as tf
import h5py
from numpy import *
from matplotlib import use
use('Agg')
from matplotlib import pyplot as pl
import argparse

#Allowing for dynamic gpu memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

def svm_net(X_train, X_test, Y_train, Y_test):
    clf_poly = SVR(C=1e3, degree=2, kernel='poly')
    clf_rbf = SVR(C=1e4, kernel='rbf')
    clf_linear = SVR(C=1e4, epsilon=0.1, kernel='linear')
    model = clf_rbf.fit(X_train, Y_train.ravel())
    score = model.predict(X_test)
    return score  

def plotter(predic, Y_test, f_title):
    #Comparing RMS pred vs. True RMS
    pl.hist(predic, bins=100, alpha=0.6, label='predicted')
    pl.hist(Y_test, bins=100, alpha=0.6, label='True')
    pl.title('Predicted and True')
    pl.legend()
    pl.savefig('/home/hunter.gabbard/public_html/Detchar/scattering/normalized/svm/comb_hist_%s.png' % f_title)
    pl.close()

    #Plotting the accuracy
    pl.hist((predic - Y_test), bins=100)
    pl.title('Accuracy')
    pl.savefig('/home/hunter.gabbard/public_html/Detchar/scattering/normalized/svm/accuracy_hist_%s.png' % f_title)
    pl.close()

    #Scater Plot of Actual as a function of Predicted
    pl.scatter(predic,Y_test)
    pl.xlim(0,1)
    pl.ylim(0,1)
    pl.plot(linspace(0,1,100), linspace(0,1,100))
    pl.title('Actual vs. Predicted')
    pl.xlabel('Predicted')
    pl.ylabel('Actual')
    pl.savefig('/home/hunter.gabbard/public_html/Detchar/scattering/normalized/svm/comb_scat_%s.png' % f_title)
    pl.close()

    #RMS plot
    pl.plot(predic, alpha=0.6, label='predicted')
    pl.plot(Y_test, alpha=0.6, label='True')
    pl.legend()
    pl.savefig('/home/hunter.gabbard/public_html/Detchar/scattering/normalized/svm/RMS_timeseries_test.png')
    pl.close()

if __name__ == '__main__':
    #construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
            help="path to input dataset")
    args = vars(ap.parse_args())

    f_title = 'test'
    t_perc = 0.55
    d = h5py.File(args['dataset'], 'r')
    X_data = d['X_data']
    Y_data = d['Y_data']
    Y_data = np.asarray(Y_data)
    X_data = np.asarray(X_data)

    idx_rm = np.where(Y_data > 400)
    X_data = np.delete(X_data, idx_rm, 0)
    Y_data = np.delete(Y_data, idx_rm, 0)
    Y_data = log(Y_data)
    Y_data = (Y_data - Y_data.min()) / (Y_data.max() - Y_data.min())

    X_train, X_test = X_data[:int(len(X_data)*t_perc),:]/X_data[:int(len(X_data)*t_perc),:].max(), X_data[int(len(X_data)*t_perc):len(X_data),:]/X_data[int(len(X_data)*t_perc):len(X_data),:].max()
    Y_train, Y_test = Y_data[:int(len(X_data)*t_perc),:], Y_data[int(len(X_data)*t_perc):len(X_data),:]

    #Running svm algorithm
    predic = svm_net(X_train, X_test, Y_train, Y_test)
    predic = predic.reshape(Y_test.shape[0],1) 
  
    #Producing results plots
    plotter(predic, Y_test, f_title) 

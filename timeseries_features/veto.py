#Script to produce veto segments for the pycbc search for compact binary coalescence signals
#Author: Hunter Gabbard
#Max Planck Institute for Gravitational Physics

import numpy as np
from gwpy.timeseries import TimeSeries
import argparse


def make_veto(predic,st,dur,thresh):
    gps_times = []
    result = predic
    result = result.reshape((result.shape[0],))
    for i in range(st+256,st+dur-256):   #-1, +2
        gps_times.append(i)
    series = TimeSeries(result, times=gps_times)
    high_scatter = series > thresh
    flag = high_scatter.to_dqflag(name='L1:DCH-HIGH_SCATTER_0_4', round=True)

    return flag

if __name__ == '__main__':
    #construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
            help="path to input data. (e.g. run_score.npy")
    ap.add_argument("-o", "--output_directory", required=True,
            help="output directory")
    ap.add_argument("-s", "--start_time", required=True,
            help="start time of lock stretch")
    ap.add_argument("-l", "--duration", required=True,
            help="duration of lock stretch")
    ap.add_argument("-t", "--threshold", required=True,
            help="Threshold above which scattering is defined (e.g. 0.4)")
    args = vars(ap.parse_args())

    #Initializing variables
    predic = np.load(args['dataset'])
    out = args['output_directory']
    st = int(args['start_time'])
    dur = int(args['duration'])

    #Getting data quality flag
    dq_flag = make_veto(predic,st,dur,float(args['threshold']))

    #Dumping flag to xml file
    dq_flag.write('%s/dqflag-%s-%s.xml' % (out,st,dur))

#Script to produce veto segments for the pycbc search for compact binary coalescence signals
#Author: Hunter Gabbard
#Max Planck Institute for Gravitational Physics

#How to combine multiple ligo dqflag xml files
#ligolw_add

import numpy as np
from gwpy.timeseries import TimeSeries
import argparse


def make_veto(predic,out,thresh):
    for fi in predic:
        print fi.split('/')[7]
        st = int(fi.split('/')[7].split('_')[1].split('-')[0])
        dur = int(fi.split('/')[7].split('_')[1].split('-')[1])
        gps_times = []
        result = np.load(fi)
        result = result.reshape((result.shape[0],))
        for i in range(st+256,st+dur-256):   #-1, +2
            gps_times.append(i)
        series = TimeSeries(result, times=gps_times)
        high_scatter = series > thresh
        flag = high_scatter.to_dqflag(name='L1:DCH-HIGH_SCATTER_0_4', round=True)

        #Dumping flag to xml file
        flag.write('%s/dqflag-%s-%s.xml' % (out,st,dur))

    return flag

if __name__ == '__main__':
    #construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, nargs='+',
            help="path to input data. (e.g. run_score.npy")
    ap.add_argument("-o", "--output_directory", required=True,
            help="output directory")
    ap.add_argument("-s", "--start_time", required=False,
            help="start time of lock stretch")
    ap.add_argument("-l", "--duration", required=False,
            help="duration of lock stretch")
    ap.add_argument("-t", "--threshold", required=True, type=float,
            help="Threshold above which scattering is defined (e.g. 0.4)")
    args = ap.parse_args()

    #Initializing variables
    predic = args.dataset
    out = args.output_directory
    #st = int(args.start_time)
    #dur = int(args.duration)

    #Getting data quality flag
    dq_flag = make_veto(predic,out,args.threshold)

    #Dumping flag to xml file
    #dq_flag.write('%s/dqflag-%s-%s.xml' % (out,st,dur))

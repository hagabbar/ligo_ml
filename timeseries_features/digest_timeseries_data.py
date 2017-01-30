#! /usr/bin/env python
import sys
import argparse

from numpy import *
from numpy.fft import fft, ifft, rfft, irfft
import scipy.signal as sig

from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from gwpy.segments import DataQualityFlag,Segment

import fir
from utils import chunk_segments, load_channels

parser = argparse.ArgumentParser(description='Generate target features for machine learning on LIGO data.')
parser.add_argument('--ifo',type=str,required=True)
parser.add_argument('-f','--segment-file',type=str)
parser.add_argument('-s','--start-time',type=int)
parser.add_argument('-e','--end-time',type=int)
parser.add_argument('-c','--channels',type=str,required=True)

args = parser.parse_args()

ifo=args.ifo

if args.segment_file:
    sci_segs=DataQualityFlag.read(args.segment_file)
    assert sci_segs.ifo == ifo
    segs=sci_segs.active
elif args.start_time and args.end_time:
    segs=[Segment(args.start_time, args.end_time)]
else:
    print "Either --segment-file, or both start and end time must be provided."
    exit(2)

st=segs[0].start
et=segs[-1].end

chan_lst=load_channels(args.channels)

chunk=16384
pad=256

full_data=[]
for chan in chan_lst:
    print 'Getting data for', chan
    position_chunks=[]
    velocity_chunks=[]
    for t1,t2 in chunk_segments(segs,chunk,pad):
        print 'Getting chunk', t1, t2
        data=TimeSeries.find(chan,t1,t2,ifo+'_R',nproc=4,verbose=True)
        tmp_pos=fir.osem_position(data,pad,new_sample_rate=1)
        tmp_vel=fir.osem_velocity(data,pad,new_sample_rate=1)
        position_chunks.append(tmp_pos.value)
        velocity_chunks.append(tmp_vel.value)
    full_data.append(concatenate(position_chunks))
    full_data.append(concatenate(velocity_chunks))
    
# Turn into a big array and dump to a file
full_data=array(full_data)
save("%s-ML-%u-%u.npy"%(ifo,st,et-st),full_data)


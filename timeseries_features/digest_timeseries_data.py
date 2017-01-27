#! /usr/bin/env python
import sys
from numpy import *
from numpy.fft import fft, ifft, rfft, irfft
import scipy.signal as sig

from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from gwpy.segments import DataQualityFlag

import fir
from utils import chunk_segments, load_channels

if __name__ == "__main__":
    segment_file=sys.argv[1]
    chan_file=sys.argv[2]
    
    sci_segs=DataQualityFlag.read(segment_file) 
    ifo=sci_segs.ifo
    segs=sci_segs.active
    
    chan_lst=load_channels(chan_file)

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


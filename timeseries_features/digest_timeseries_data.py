#! /usr/bin/env python
import sys
from numpy import *
from numpy.fft import fft, ifft, rfft, irfft
import scipy.signal as sig

from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from gwpy.segments import DataQualityFlag

import fir

def get_science_segments(ifo, st, et, min_length=1800):
    scimode=DataQualityFlag.query(ifo+':DMT-ANALYSIS_READY:1',st,et)
    print scimode.known
    print scimode.active
    bad_segs=[flg for flg in scimode.active if abs(flg) < min_length]
    for flg in bad_segs:
        scimode.active.remove(flg)
    return scimode.active

def load_channels(channel_file):
    with open(channel_file) as ff:
        channel_list=[line.split()[0] for line in ff]
    return channel_list

def chunk_segments(seg_lst, chunk_size, pad_size):
    """Split a segment list into chunks, respecting padding"""
    assert chunk_size > 2*pad_size
    for seg in seg_lst:
        t1,t2=int(seg.start),int(seg.end)
        duration=t2-t1
        for tt in xrange(0, duration-2*pad_size, chunk_size-2*pad_size):
            if tt+chunk_size<duration:
                yield (t1+tt,t1+tt+chunk_size)
            elif tt+2*pad_size<duration:
                yield (t1+tt,t1+duration)

if __name__ == "__main__":
    ifo=sys.argv[1]
    st=int(sys.argv[2])
    et=int(sys.argv[3])
    chan_file=sys.argv[4]

    chunk=16384
    pad=256

    target_chan=ifo+':GDS-CALIB_STRAIN'

    chan_lst=load_channels(chan_file)

    segs=get_science_segments(ifo,st,et)
    for seg in segs:
        print seg.start, seg.end


    full_data=[]
    for idx,chan in enumerate(chan_lst):
        print 'Getting data for', chan
        if idx == 0:
            subsys_orig = chan.split(':')[1].split('_')[0]
        elif idx > 0:
            subsys_new = chan.split(':')[1].split('_')[0]
        if subsys_new == subsys_orig:
            position_chunks=[]
            velocity_chunks=[]
            for t1,t2 in chunk_segments(segs,chunk,pad):
                print 'Getting chunk', t1, t2
                data=TimeSeries.find(chan,t1,t2,ifo+'_R',nproc=20,verbose=True)
                tmp_pos=fir.osem_position(data,pad,new_sample_rate=1)
                tmp_vel=fir.osem_velocity(data,pad,new_sample_rate=1)
                position_chunks.append(tmp_pos.value)
                velocity_chunks.append(tmp_vel.value)
            full_data.append(concatenate(position_chunks))
            full_data.append(concatenate(velocity_chunks))
            subsys_orig = chan.split(':')[1].split('_')[0]
        elif subsys_new =! subsys_orig:
            # Turn into a big array and dump to a file
            full_data=array(full_data)
            save("%s-%s-ML-%u-%u.npy"%(ifo,subsys_orig,st,et-st),full_data)
            full_data = []
            for t1,t2 in chunk_segments(segs,chunk,pad):
                print 'Getting chunk', t1, t2
                data=TimeSeries.find(chan,t1,t2,ifo+'_R',nproc=20,verbose=True)
                tmp_pos=fir.osem_position(data,pad,new_sample_rate=1)
                tmp_vel=fir.osem_velocity(data,pad,new_sample_rate=1)
                position_chunks.append(tmp_pos.value)
                velocity_chunks.append(tmp_vel.value)
            full_data.append(concatenate(position_chunks))
            full_data.append(concatenate(velocity_chunks))
            subsys_orig = chan.split(':')[1].split('_')[0]
    # Turn into a big array and dump to a file
    full_data=array(full_data)
    save("%s-%s-ML-%u-%u.npy"%(ifo,subsys_new,st,et-st),full_data)



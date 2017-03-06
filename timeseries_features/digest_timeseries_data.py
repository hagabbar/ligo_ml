#! /usr/bin/env python
import sys
import argparse

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
    subsys_new=''
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
        elif subsys_new != subsys_orig:
            # Turn into a big array and dump to a file
            full_data=array(full_data)
            save("%s-%s-ML-%u-%u.npy"%(ifo,subsys_orig,st,et-st),full_data)
            full_data = []
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
    # Turn into a big array and dump to a file
    full_data=array(full_data)
    save("%s-%s-ML-%u-%u.npy"%(ifo,subsys_new,st,et-st),full_data)



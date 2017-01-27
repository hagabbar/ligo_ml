#! /usr/bin/env python
import sys
from numpy import *
from numpy.fft import fft, ifft, rfft, irfft
import scipy.signal as sig
import numpy as np

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
    range_q='(4,64)'

    target_chan=ifo+':GDS-CALIB_STRAIN'

    chan_lst=load_channels(chan_file)

    segs=get_science_segments(ifo,st,et)
    for seg in segs:
        print seg.start, seg.end


    full_data=[]
    darm_data=[]
    for chan in chan_lst:
        print 'Getting data for', chan
        position_chunks=[]
        velocity_chunks=[]
        q_chunks=[]
        for t1,t2 in chunk_segments(segs,chunk,pad):
            print 'Getting chunk', t1, t2
            data=TimeSeries.find(chan,t1,t2,ifo+'_R',nproc=6,verbose=True)
            tmp_pos=fir.osem_position(data,pad,new_sample_rate=1)
            tmp_vel=fir.osem_velocity(data,pad,new_sample_rate=1)
            position_chunks.append(tmp_pos.value)
            velocity_chunks.append(tmp_vel.value)

            #Calculate q_transform chunks
            tmp_q=fir.q_scan(data,pad,new_sample_rate=1)
            q_tstep=round(tmp_q.shape[0]/float(len(data))) * data.sample_rate.value
            for q_v in range(0+(pad*1000),len(tmp_q)-(pad*1000),1000):
               if q_v == 0+(pad*1000):
                   small_chunksq = tmp_q[q_v,:]
                   small_chunksq = np.asarray(small_chunksq)   
               else: 
                   small_chunksq = np.vstack((small_chunksq,tmp_q[q_v,:]))                             
            q_chunks.append(small_chunksq) 
        full_data.append(concatenate(position_chunks))
        full_data.append(concatenate(velocity_chunks))
        full_data.append(concatenate(q_chunks))
        sys.exit()        

        # Get DARM BLRMS
        chunk=4096
        srate=16384.
        filt=sig.firwin(int(2*pad*srate),[50.,59.9,60.1,100.],nyq=srate/2.,window='hann',pass_zero=False)
    darm_blrms_chunks=[]
    for t1,t2 in chunk_segments(segs,chunk,pad):
        data=TimeSeries.get(target_chan,t1-1,t2,nproc=6,verbose=True)
        assert data.sample_rate.value==srate
        tmp_bp=1.e21*fir.apply_fir(data, filt, new_sample_rate=256, deriv=False)
        darm_blrms_chunks.append(tmp_bp[128:-128].rms(1.).value)
    darm_data.append(concatenate(darm_blrms_chunks))
    
    # Turn into a big array and dump to a file
    full_data=array(full_data)
    darm_data=array(darm_data)
    save('%s-ML-%s-%s.npy' % (ifo,st,et-st),full_data)
    save('%s-DARMBLRMS-%s-%s.npy' % (ifo,st,et-st),darm_data)


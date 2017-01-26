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

    chunk=16384
    pad=256

    target_chan=ifo+':GDS-CALIB_STRAIN'

    segs=get_science_segments(ifo,st,et)
    for seg in segs:
        print seg.start, seg.end
       
    
    # Get DARM BLRMS
    chunk=4096
    srate=16384.
    filt=sig.firwin(int(2*pad*srate),[25.,59.9],nyq=srate/2.,window='hann',pass_zero=False)
    darm_blrms_chunks=[]
    for t1,t2 in chunk_segments(segs,chunk,pad):
        data=TimeSeries.get(target_chan,t1-1,t2,nproc=6,verbose=True)
        assert data.sample_rate.value==srate
        tmp_bp=1.e21*fir.apply_fir(data, filt, new_sample_rate=256, deriv=False)
        darm_blrms_chunks.append(tmp_bp[128:-128].rms(1.).value)
   
    # Turn into a big array and dump to a file
    full_data=array(concatenate(darm_blrms_chunks))
    save("%s-DARMBLRMS-%u-%u.npy"%(ifo,st,et-st),full_data)


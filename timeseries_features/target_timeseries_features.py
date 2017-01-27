#! /usr/bin/env python
import sys
from numpy import *
from numpy.fft import fft, ifft, rfft, irfft
import scipy.signal as sig

from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from gwpy.segments import DataQualityFlag

import fir
from utils import chunk_segments

if __name__ == "__main__":
    segment_file=sys.argv[1]
    sci_segs=DataQualityFlag.read(segment_file)
    
    ifo=sci_segs.ifo
    segs=sci_segs.active

    chunk=4096
    pad=256

    target_chan=ifo+':GDS-CALIB_STRAIN'
    
    # Get DARM BLRMS
    srate=16384.
    filt=sig.firwin(int(2*pad*srate),[25.,59.9],nyq=srate/2.,window='hann',pass_zero=False)
    darm_blrms_chunks=[]
    for t1,t2 in chunk_segments(segs,chunk,pad):
        print 'Getting chunk', t1, t2
        data=TimeSeries.get(target_chan,t1-1,t2,nproc=6,verbose=True)
        assert data.sample_rate.value==srate
        tmp_bp=1.e21*fir.apply_fir(data, filt, new_sample_rate=256, deriv=False)
        darm_blrms_chunks.append(tmp_bp[128:-128].rms(1.).value)
   
    # Turn into a big array and dump to a file
    full_data=array(concatenate(darm_blrms_chunks))
    save("%s-DARMBLRMS-%u-%u.npy"%(ifo,st,et-st),full_data)


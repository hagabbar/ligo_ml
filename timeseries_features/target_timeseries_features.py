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
from utils import chunk_segments

parser = argparse.ArgumentParser(description='Generate target features for machine learning on LIGO data.')
parser.add_argument('--ifo',type=str,required=True)
parser.add_argument('-f','--segment-file',type=str)
parser.add_argument('-s','--start-time',type=int)
parser.add_argument('-e','--end-time',type=int)

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

chunk=4096
pad=256

target_chan=ifo+':GDS-CALIB_STRAIN'
frtype=ifo+"_HOFT_C00"

# Get DARM BLRMS
srate=16384.
filt=sig.firwin(int(2*pad*srate),[25.,59.9],nyq=srate/2.,window='hann',pass_zero=False)
darm_blrms_chunks=[]
for t1,t2 in chunk_segments(segs,chunk,pad):
    print 'Getting chunk', t1, t2
    data=TimeSeries.get(target_chan,t1-1,t2,frtype,nproc=4,verbose=True)
    assert data.sample_rate.value==srate
    tmp_bp=1.e21*fir.apply_fir(data, filt, new_sample_rate=256, deriv=False)
    darm_blrms_chunks.append(tmp_bp[128:-128].rms(1.).value)

# Turn into a big array and dump to a file
full_data=array(concatenate(darm_blrms_chunks))
save("%s-DARMBLRMS-%u-%u.npy"%(ifo,st,et-st),full_data)


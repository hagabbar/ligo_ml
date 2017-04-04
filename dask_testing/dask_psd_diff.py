#!/usr/bin/env python

from distributed import Client
from gwpy.io import datafind
import sys
import argparse
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from gwpy.segments import DataQualityFlag,Segment
from numpy import sum, abs
from numpy import *
from numpy.fft import fft, ifft, rfft, irfft
import scipy.signal as sig
import fir
import pickle

def find_raw_frames(ifo,start,end):
    try:
        conn = datafind.connect()
        cache = conn.find_frame_urls(ifo[0], ifo+'_R',
                                    start, end, urltype='file')
    except RuntimeError:
        conn.close()
        cache = conn.find_frame_urls(ifo[0], ifo+'_R',
                                    start, end, urltype='file')
    return cache

def aux_feat_get(params):
    import sys
    sys.path.append('/home/hunter.gabbard/Detchar/ligo_ml/timeseries_features')
    import fir
    chan,ifo=params[0],params[1]
    print chan
    full_data=[]
    position_chunks=[]
    velocity_chunks=[]
    try:
        data=TimeSeries.find(chan,t1,t2,ifo+'_R',verbose=True)
        tmp_pos=fir.osem_position(data,pad,new_sample_rate=16)
        tmp_vel=fir.osem_velocity(data,pad,new_sample_rate=16)
        position_chunks.append(tmp_pos.value)
        velocity_chunks.append(tmp_vel.value)
        full_data.append(concatenate(position_chunks))
        full_data.append(concatenate(velocity_chunks))
    except RuntimeError as err:
        return err
    full_data=array(full_data)
    return (chan,full_data)

def get_darm_blrms(pad,segs,chunk,target_chan,frtype):
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

    return darm_blrms_chunks

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

#Add in argument parser
parser = argparse.ArgumentParser(description='Compute comparison PSDs between two times for many channels.')
parser.add_argument('-c','--channels', type=str, help='File containing channel names', required=True)
parser.add_argument('-a','--address', type=str, help='Address of Dask scheduler, e.g. 127.0.0.1:8786', required=True)
parser.add_argument('--sci-seg', help='hdf file of science segments you wish to run over', required=False)
parser.add_argument('-s', '--start-time', type=int, help='GPS start time', required=False)
parser.add_argument('-e','--end-time',type=int, help='GPS end time', required=False)
parser.add_argument('-i', '--ifo', type=str, help='observatory', required=False)

args = parser.parse_args()
#Uncomment in future if you want to do something with psd comparison tool
#if len(args.start_psd) != 2:
#    print "Requires exactly two start times."
#    sys.exit()

# Load channel list; accept FrChannels format
channels = []
with open(args.channels,'r') as fin:
    for line in fin:
        channels.append(line.split()[0])


# Shortcuts for the times
#Uncomment in future if you want to do something with psd comparison tool
#st1,st2 = min(args.start_psd),max(args.start_psd)
if args.duration:
    dur = args.duration

#Get ifo and active sci segs
#ifo=sci_segs.ifo
ifo=args.ifo
if args.sci_seg:
    sci_segs=DataQualityFlag.read(args.sci_seg, path='H1:DMT-ANALYSIS_READY:1')
    assert sci_segs.ifo == ifo
    segs=sci_segs.active
elif args.start_time and args.end_time:
    segs=[Segment(args.start_time, args.end_time)]
else:
    print "Either --segment-file, or both start and end time must be provided."
    sys.exit(2)

st=segs[0].start
et=segs[-1].end

###################
###DARM Channels###
###################

target_chan=ifo+':GDS-CALIB_STRAIN'
frtype=ifo+"_HOFT_C00"

chunk=4096
pad=256

darm_blrms_chunks = get_darm_blrms(pad,segs,chunk,target_chan,frtype)

# Turn into a big array and dump to a file
full_data=array(concatenate(darm_blrms_chunks))
save("%s-DARMBLRMS-%u-%u.npy"%(ifo,st,et-st),full_data)

##################
###Aux channels###
##################

chunk=16384
pad=256

# Find the data
#cache1=find_raw_frames(ifo, st1, st1+dur)
#cache2=find_raw_frames(ifo, st2, st2+dur)

# Connect to Dask scheduler
client = Client(args.address)

for t1,t2 in chunk_segments(segs,chunk,pad):
    print 'Getting chunk', t1, t2

    # Set up the channel list 
    params_list = [(chan,ifo,t1,t2) for chan in channels] #Add in st1, st2, dur for psd comparison tool

    # Run jobs on the cluster and return results
    jobs = client.map(aux_feat_get, params_list)
    print len([x for x in jobs if x.status=='finished'])
    result = client.gather(jobs)

    # Write out the results
    #Will sort the results by how much difference in the PSD there is
    #result.sort(key=lambda x: x[1], reverse=True)

    print result

    with open('results_of_aux_%u-%u.dat' % (t1,(t2-t1)),'wb') as fout:
        pickle.dump(result, fout)

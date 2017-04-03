#!/usr/bin/env python

from distributed import Client
from gwpy.io import datafind

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

import argparse
parser = argparse.ArgumentParser(description='Compute comparison PSDs between two times for many channels.')
parser.add_argument('-c','--channels', type=str, help='File containing channel names', required=True)
parser.add_argument('-a','--address', type=str, help='Address of Dask scheduler, e.g. 127.0.0.1:8786', required=True)
parser.add_argument('-s','--start', type=int, action='append', help='GPS start time.', required=True)
parser.add_argument('-d','--duration', type=int, help='Duration (sec)', required=True)

args = parser.parse_args()
if len(args.start) != 2:
    print "Requires exactly two start times."
    exit()

# Load channel list; accept FrChannels format
channels = []
with open(args.channels,'r') as fin:
    for line in fin:
        channels.append(line.split()[0])


# Shortcuts for the times
st1,st2 = min(args.start),max(args.start)
dur = args.duration

# Set up the channel list 
params_list = [(chan,st1,st2,dur) for chan in channels]

# Find the data
ifo=channels[0].split(':')[0]
cache1=find_raw_frames(ifo, st1, st1+dur)
cache2=find_raw_frames(ifo, st2, st2+dur)

def psd_diff_helper(params):
    from gwpy.timeseries import TimeSeries
    from numpy import sum, abs
    chan=params[0]
    #st1,st2,dur=params[1:4]
    
    # Get data and discard units since some channels have anomalous units
    try:
        data1=TimeSeries.read(cache1, chan, st1, st1+dur)
        #data1.override_unit('')
        data2=TimeSeries.read(cache2, chan, st2, st2+dur)
        #data2.override_unit('')
    except RuntimeError as err:
        return err
    
    psd1=data1.psd(4,2,method='median').value[40:1600]
    psd2=data2.psd(4,2,method='median').value[40:1600]
    if psd1.min()==0. or psd2.min()==0.:
        return (chan,-1)
    result = sum(abs(psd2-psd1)*(1./psd1+1./psd2))
    return (chan, result)



# Connect to Dask scheduler
client = Client(args.address)

# Run jobs on the cluster and return results
jobs = client.map(psd_diff_helper, params_list)
result = client.gather(jobs)

# Write out the results
result.sort(key=lambda x: x[1], reverse=True)
with open('results_of_psd_diff.txt','w') as fout:
    fout.write('# %u %u %u' % (st1,st2,dur))
    for line in result:
        fout.write('%s %.4g\n' % line)


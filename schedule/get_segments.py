#! /usr/bin/env python
import sys

from gwpy.segments import DataQualityFlag

def get_science_segments(ifo, st, et, min_length=1800):
    """Get observation ready time, removing segments that are too short.
    Remove one second from odd-length segments (helps with filtering.
    
    Returns: gwpy.segments.DataQualityFlag""" 
    scimode=DataQualityFlag.query(ifo+':DMT-ANALYSIS_READY:1',st,et)
    print scimode.known
    print scimode.active
    bad_segs=[flg for flg in scimode.active if abs(flg) <= min_length]
    for flg in bad_segs:
        scimode.active.remove(flg)
    # Force all segments to be even length
    for idx,flg in enumerate(scimode.active):
        if not abs(flg) % 2 == 0:
            scimode.active[idx]=type(flg)(flg.start,flg.end-1)
    return scimode

if __name__ == "__main__":
    ifo=sys.argv[1]
    st=int(sys.argv[2])
    et=int(sys.argv[3])

    scimode = get_science_segments(ifo, st, et)
    scimode.write('%s-SCISEGS-%u-%u.hdf'%(ifo,st,et-st))


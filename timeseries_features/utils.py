from gwpy.segments import DataQualityFlag

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


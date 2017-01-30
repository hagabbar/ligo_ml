#! /usr/bin/env python
import sys

from gwpy.segments import DataQualityFlag

def add_job(the_file, job_type, job_number, **kwargs):
    job_id="%s%.6u" % (job_type, job_number)
    the_file.write("JOB %s %s.sub\n" % (job_id, job_type))
    vars_line=" ".join(['%s="%s"'%(arg,str(val))
                            for arg,val in kwargs.iteritems()])
    the_file.write("VARS %s %s\n" % (job_id, vars_line))
    the_file.write("\n")

if __name__ == "__main__":
    segment_file=sys.argv[1]
    sci_segs=DataQualityFlag.read(segment_file)

    ifo=sci_segs.ifo
    segs=sci_segs.active

    jobtypes=['target','aux']

    fdag = open("my.dag",'w')
    for idx, seg in enumerate(segs):
        for jobtype in jobtypes:
            add_job(fdag, jobtype, idx, ifo=ifo, st=seg.start, et=seg.end)


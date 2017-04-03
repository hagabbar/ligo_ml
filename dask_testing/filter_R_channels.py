#! /usr/bin/env python
import sys

reject_subsys=['CAL-','ALS-','OMC-']
reject_keywords=['DARM','DCPD','MASTER','NOISEMON','ODC','_EXC_DQ','PI_']

def accept_chan(chan, srate):
    if srate < 1024:
        return False
    for subsys in reject_subsys:
        if subsys in chan:
            return False
    for keyword in reject_keywords:
        if keyword in chan:
            return False
    return True

with open(sys.argv[1],'r') as fin:
    for line in fin:
        pieces = line.split()
        chan,srate=pieces[0], int(pieces[1])
        if accept_chan(chan, srate):
            print chan


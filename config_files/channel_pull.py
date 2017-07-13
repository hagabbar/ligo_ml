#Script to pull both top stage osems and lower stage osem channel names and sampling frequencies
#Autor: Hunter G.

import sys
import os
import numpy as np

def get_chan(optics):
    for op in optics:
        if op == ['OM1','OM2','OM3','OMC']:
            f = file.open('channels.txt')
            f.write('SUS-%s_'


def main():
optics = ['OM1','OM2','OM3','OMC','RM1','RM2','ITMX','ITMY','BS','MC1','MC3','PRM','PR3','MC2','PR2','SR2','SR3','SRM','ETMX','ETMY','TMSX','TSMY','IM1','IM2','IM3','IM4']


if __name__ == '__main__':
    main()

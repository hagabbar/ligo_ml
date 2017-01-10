from gwpy.timeseries import TimeSeries
from numpy import *
from numpy.fft import rfft,irfft
import scipy.signal as sig
from fir import fir_helper
import pandas as pd
import numpy as np
import h5py
import sys
from gwpy.segments import DataQualityFlag, Segment

def load_data(ifo, chan, chan_code, st, offset): #Last variable is offset
    full_chan='%s:%s'%(ifo,chan)
    print full_chan
    data_full=[]
    
    if chan == 'GDS-CALIB_STRAIN':
        end_time = st+offset+chunk_size
        if end_time > st+seg_length:
            diff = end_time - (st+seg_length)
            data_full=TimeSeries.get(full_chan,st+offset,st+offset+chunk_size - diff,verbose=True,nproc=8)
        else:
            data_full=TimeSeries.get(full_chan,st+offset,st+(offset)+chunk_size,verbose=True,nproc=8)
    else:
        data_full=TimeSeries.get(full_chan,st,st+(offset),verbose=True)
    return data_full

def diriv_filt(data):
    result=fir_helper(data,
                        [0.,0.01,0.02,0.4,7.,8.,128.],
                        [0.,0.,2*pi*0.02,2*pi*0.4,2*pi*7.,1.e-4,0.],
                        pad_sec=64,
                        new_sample_rate=None, deriv=True)
    return result


if __name__ == '__main__':
    ifo = 'L1'
    st = 1164686978
    #Edit botht the duration/seg length and st. Start time 1164326417
    dur= 24005
    pad=64
    chunk_size = 4096
    aux_chans = ['SUS-ETMY_M0_DAMP_L_IN1_DQ','SUS-ETMY_M0_DAMP_P_IN1_DQ','SUS-ETMY_M0_DAMP_Y_IN1_DQ','SUS-ETMY_R0_DAMP_L_IN1_DQ',
                'SUS-ETMY_R0_DAMP_P_IN1_DQ','SUS-ETMY_R0_DAMP_Y_IN1_DQ','SUS-TMSY_M1_DAMP_L_IN1_DQ','SUS-TMSY_M1_DAMP_P_IN1_DQ',
                'SUS-TMSY_M1_DAMP_Y_IN1_DQ','SUS-ETMX_M0_DAMP_L_IN1_DQ','SUS-ETMX_M0_DAMP_P_IN1_DQ','SUS-ETMX_M0_DAMP_Y_IN1_DQ',
                'SUS-ETMX_R0_DAMP_L_IN1_DQ','SUS-ETMX_R0_DAMP_P_IN1_DQ','SUS-ETMX_R0_DAMP_Y_IN1_DQ','SUS-ITMY_M0_DAMP_L_IN1_DQ',
                'SUS-ITMY_M0_DAMP_P_IN1_DQ','SUS-ITMY_M0_DAMP_Y_IN1_DQ','SUS-ITMY_R0_DAMP_L_IN1_DQ','SUS-ITMY_R0_DAMP_P_IN1_DQ',
                'SUS-ITMY_R0_DAMP_Y_IN1_DQ','SUS-ITMX_M0_DAMP_L_IN1_DQ','SUS-ITMX_M0_DAMP_P_IN1_DQ','SUS-ITMX_M0_DAMP_Y_IN1_DQ',
                'SUS-ITMX_R0_DAMP_L_IN1_DQ','SUS-ITMX_R0_DAMP_P_IN1_DQ','SUS-ITMX_R0_DAMP_Y_IN1_DQ']  
    data_all = {}  
    data_aux = {}
    data_darm = {}
    X_data = [] 
    np.asarray(X_data)

    #Retrieving Science Active Times
    dqsci=DataQualityFlag.query('H1:DMT-ANALYSIS_READY:1',st,st+dur)
    for flg in dqsci.active:
        flg_start = int(flg.start)
        seg_length = int(abs(flg))
        flg_end = int(flg.start) + int(abs(flg))
 
        #Calculating the band-passed rms for DARM
        for offset in range(0,seg_length,chunk_size-(pad*2)):
            data = load_data(ifo,'GDS-CALIB_STRAIN','HOFT',flg_start,offset)

            #Computing the spectrogram from DARM timeseries
            print 'Computing Spectrogram of DARM'
            f, t, Sxx = sig.spectrogram(data)

            nyquist=data.sample_rate.value/2.
            if data.duration.value <= 2*pad:
                continue
            data_bp=fir_helper(data,
                                [0.,25.,30.,42.1,42.3,42.5,59.8,60.,60.2,120,125,nyquist],
                                [0.,1.e-6,1.,1.,1.e-2,1.,1.,1.e-2,1,1,1.e-6,0],
                                pad_sec=pad)
        
            print int(data_bp.epoch.gps)
            if offset == 0:
                data_bp_final = data_bp.rms(1).copy()
                f_f, t_f, Sxx_f = f, t, Sxx
            else:
                data_bp_final.append(data_bp.rms(1).copy())
                f_f, t_f, Sxx_f = f_f.append(f), t_f.append(t), Sxx_f.append(Sxx)
    Y_data = np.asarray(data_bp_final).reshape(data_bp_final.shape[0],1)  

    data_darm['GDS-CALIB_STRAIN'] = data_bp_final  
    del data     

    sys.exit()
    #Calculating the rms values and velocities of the auxiliary channels
    idx = 0
    for ch in aux_chans:
        #Retrieving Science Active Times
        dqsci=DataQualityFlag.query('H1:DMT-ANALYSIS_READY:1',st,st+dur)
        idx+=1
        for flg in dqsci.active:
            flg_start = int(flg.start)
            seg_length = int(abs(flg))
            flg_end = int(flg.start) + int(abs(flg))

            data_full = load_data(ifo,ch,'HOFT',flg_start,seg_length)
            for offset in range(0,seg_length,chunk_size-(pad*2)):
                #data = load_data(ifo,ch,'HOFT',st,offset)
                data = data_full[offset*data_full.sample_rate.value:offset*data_full.sample_rate.value+chunk_size*data_full.sample_rate.value].copy()
                nyquist=data.sample_rate.value/2.
                if data.duration.value <= 2*pad:
                    continue

                #Calculating the velocities of the aux channels
                data_v = diriv_filt(data)
            
                #Band-passing and taking the rms of time series
                data_bp=fir_helper(data,
                                    [0.,25.,30.,42.1,42.3,42.5,59.8,60.,60.2,120,125,nyquist],
                                    [0.,1.e-6,1.,1.,1.e-2,1.,1.,1.e-2,1,1,1.e-6,0],
                                    pad_sec=pad)
            
                #Subtracting off mean and dividing by standard deviation
                        

                print int(data_bp.epoch.gps)
                if offset == 0:
                    data_v_final = data_v.rms(1).copy()
                    data_bp_aux_f= data.rms(1)[pad:-pad].copy()
                else:
                    data_v_final.append(data_v.rms(1)).copy()
                    data_bp_aux_f.append(data.rms(1)[pad:-pad].copy())
        data_bp_norm = (data_full.rms(1)[pad:-pad] - data_full.mean().value) / data_full.std().value

        #Storing info for machine learning
        data_v_final = np.asarray(data_v_final).reshape(data_v_final.shape[0],1)
        data_bp_aux_f = np.asarray(data_bp_aux_f).reshape(data_bp_aux_f.shape[0],1)
        data_bp_norm = np.asarray(data_bp_norm).reshape(data_bp_norm.shape[0],1)
        aux_comb_data = np.hstack((data_v_final,data_bp_aux_f,data_bp_norm))         
        if idx == 1:
            X_data = aux_comb_data
        else:
            X_data = np.hstack((X_data, aux_comb_data))

        data_aux[ch+'_v'] = data_v_final
        data_aux[ch+'_bprms'] = data_bp_aux_f
        data_aux[ch+'_norm'] = data_bp_norm

    #Writing data to file for later use on dgx-1 machine
    with h5py.File('data.hdf', 'w') as hf:
        #hf.create_dataset('data_aux', data=data_aux)
        #hf.create_dataset('data_darm', data=data_darm)
        hf.create_dataset('X_data', data=X_data)
        hf.create_dataset('Y_data', data=Y_data)
    

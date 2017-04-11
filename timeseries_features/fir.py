from gwpy.timeseries import TimeSeries
from numpy import *
from numpy.fft import rfft,irfft
import scipy.signal as sig

def apply_fir(data, taps, new_sample_rate=None, deriv=False):
    """Apply an FIR filter to a gwpy.TimeSeries, using the FFT.
    The FIR filter is specified as time-domain coefficients (taps)
    and is forced to be zero-phase
    Downsampling can be done, but the user is responsible for low-passing the data."""
    tfilt=zeros(len(data),dtype=data.dtype)
    tfilt[:len(taps)]=taps
    ffilt=abs(rfft(tfilt)) # Abs enforces zero phase
    temp=rfft(data.value)*ffilt
    if deriv: # If taking the derivative, need a 90 degree phase
        temp=1.j*temp

    pad_sec=len(taps)/(2.*data.sample_rate.value) # Length of filter corruption on either side
    if new_sample_rate:
        temp=(new_sample_rate/data.sample_rate.value)*temp[:1+int(data.duration.value*new_sample_rate/2)]
    srate=new_sample_rate or data.sample_rate.value
    valid_idx=int(pad_sec*srate)
    return TimeSeries(irfft(temp)[valid_idx:-valid_idx], sample_rate=srate,
                        name=data.name, channel=data.channel,
                        epoch=data.epoch.gps+pad_sec)

def fir_helper(data, freqs, resps, pad_sec, new_sample_rate=None, deriv=False):
    srate=data.sample_rate.value 
    print srate
    print freqs
    filt=sig.firwin2(int(2*pad_sec*srate),
                        freqs, resps,
                        window='hann',nyq=srate/2.)
 
    return apply_fir(data,filt,new_sample_rate=new_sample_rate,deriv=deriv)

def osem_position(osem_data, pad_sec=64, new_sample_rate=None):
    nyquist=osem_data.sample_rate.value/2.
    assert new_sample_rate >= 16
    result = fir_helper(osem_data,
                        [0.,0.04,0.08,7.2,nyquist],
                        [0.,1.e-4,1.,1.,0.],
                        pad_sec=pad_sec,
                        new_sample_rate=16, deriv=False) #set back new_sample_rate = new_sample_rate when done debugging
    return result

def osem_velocity(osem_data, pad_sec=64, new_sample_rate=None):
    nyquist=osem_data.sample_rate.value/2.
    assert new_sample_rate >= 16
    result = fir_helper(osem_data,
                        [0.,0.04,0.08,7.2,nyquist],
                        [0.,1.e-4,2.*pi*0.08,2.*pi*7.2,0.],
                        pad_sec=pad_sec,
                        new_sample_rate=new_sample_rate, deriv=True)
    return result

if __name__ == '__main__':
    times=arange(1024*256,dtype=float32)/256.
    test=TimeSeries(cos(2.*pi*0.4*times)+2.*sin(2.*pi*1.*times)-cos(2.*pi*3.*times)-10.*sin(2.*pi*10.*times),sample_rate=256,name='test')
    dtest=TimeSeries(2.*pi*(-0.4*sin(2.*pi*0.4*times)+2.*cos(2.*pi*1.*times)+3.*sin(2.*pi*3.*times)), sample_rate=256,name='deriv')

    result1=fir_helper(test,
                        [0.,0.01,0.02,0.4,7.,8.,128.],
                        [0.,0.,2*pi*0.02,2*pi*0.4,2*pi*7.,1.e-4,0.],
                        pad_sec=64)

    result=fir_helper(test,
                        [0.,0.01,0.02,0.4,7.,8.,128.],
                        [0.,0.,2*pi*0.02,2*pi*0.4,2*pi*7.,1.e-4,0.],
                        pad_sec=64,
                        new_sample_rate=64, deriv=True)

    p1=dtest[64*256:(64+5)*256].plot(label='a')
    p1.gca().plot(result[:5*64],label='b')
    p1.gca().legend()
    p1.show()



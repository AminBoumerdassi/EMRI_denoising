"""
Functions for analysis EMRIs after they have been generated
"""

import numpy as np
import cupy as cp
from scipy.signal.windows import tukey

#Few imports
from few.trajectory.inspiral import EMRIInspiral
from few.utils.geodesic import get_separatrix
from few.utils.utility import get_p_at_t
from few.utils.constants import YRSID_SI


#LISA tools imports
from lisatools.sensitivity import get_sensitivity, A1TDISens, T1TDISens
from lisatools.detector import EqualArmlengthOrbits, ESAOrbits
from lisatools.utils.constants import lisaLT

#fast lisa response imports
from fastlisaresponse import ResponseWrapper

def zero_pad(data):
    """
    Zero pads an array so its length is a power of two.
    """
    #Check whether to use cupy or numpy
    xp = cp.get_array_module(data)
    N = len(data)
    pow_2 = xp.ceil(xp.log2(N))
    return xp.pad(data,(0,int((2**pow_2)-N)),'constant')
def zero_pad_length(data):
    """
    Returns the length of an array when padded to have length
    of the next power of 2.
    """
    xp= cp.get_array_module(data)
    #Convert input to np array if not already
    data= xp.array(data)
    N = data.shape[-1]
    pow_2 = xp.ceil(xp.log2(N))
    pad_width= int((2**pow_2))
    return pad_width
def zero_pad_BATCHWISE(data):
    """
    Zero-pads a batch of vectors to length 2^x.
    Input: (batch_size, no. channels, vector_length)
    Output: (batch_size, no. channels, padded_vec_length)
    """
    #Check whether to use cupy or numpy
    xp = cp.get_array_module(data)
    N = data.shape[2]#len(data)
    pow_2 = xp.ceil(xp.log2(N))
    pad_width= ((0,0),(0,0),(0,int((2**pow_2)-N)))
    return xp.pad(data, pad_width, 'constant')
def noise_PSD_AE(N_t, delta_t, TDI = 'TDI1', include_foreground=True, model='scirdv1', xp=np):
    """
    Takes in frequency, spits out TDI1 or TDI2 A channel, same as E channel is equal and constant arm length approx. 
    """
    #xp= cp.get_array_module(f)
    #Get frequency bins for the corresponding length of time
    f = xp.fft.rfftfreq(N_t, delta_t)
    f[0] = f[1]   # To "retain" the zeroth frequency
    #Start with the TDI1 PSD, then add the leftover terms for TDI2 if needed
    if include_foreground:
        kwargs = {"stochastic_params" : (N_t*delta_t,)}
    else:
        kwargs = {}
    S = get_sensitivity(f, sens_fn=A1TDISens, model=model, return_type='PSD', **kwargs)
    if TDI == "TDI2":
        x = 2.0 * xp.pi * lisaLT * f
        tdi_factor = 4 * xp.sin(2*x)**2
        #Sn = sens*tdi_factor
        S *= tdi_factor
    #Clip the PSD so that the minimum possible value is the 1st one
    S[S<S[0]] = S[0]
    return xp.array([S,S])
def noise_td_AET(N, dt, PSD, return_cupy=True):# channels=["AE","AE","T"],
    """ 
    Generate time-domain noise coloured by the AET channel PSDs.
    """
    #Check whether to use cupy or numpy
    if return_cupy==True:
        xp=cp
    else:
        xp=np
    # #Extract frequency bins for use in PSD
    N_padded= len(zero_pad(xp.ones(N)))
    # freq = xp.fft.rfftfreq(N_padded , dt)
    # freq[0] = freq[1]#avoids NaNs in PSD[0]
    # PSD_AET= xp.asarray([get_sensitivity(freq, sens_fn=A1TDISens, return_type="PSD") for channel in channels])
    #Draw samples from multivariate Gaussian
    variance_noise_f= 0.25* N_padded*dt*PSD#N*PSD_AET/(4*dt)#should we use N_padded rather than N?
    noise_f = xp.random.normal(0,xp.sqrt(variance_noise_f)) + 1j*xp.random.normal(0,xp.sqrt(variance_noise_f))
    #Transforming the frequency domain signal into the time domain
    return xp.fft.irfft(noise_f, n=N)#[:,:N]
def noise_td_AET_BATCHWISE(N, dt, batch_size, channels=["AE","AE"], return_cupy=True):
    """
    Generate batches of TD AET noise.
    output: (batch_size, no. channels, time_steps) 
    """
    #Check whether to use cupy or numpy
    if return_cupy==True:
        xp=cp
    else:
        xp=np
    #Pad N to nearest power of 2 for faster FFT calculation
    N_padded= len(zero_pad(xp.ones(N)))
    #Extract frequency bins for use in PSD
    freq = xp.fft.rfftfreq(N_padded, dt)
    freq[0] = freq[1]#avoids NaNs in PSD[0]
    PSD_AET= xp.asarray([get_sensitivity(freq, sens_fn=A1TDISens, return_type="PSD") for channel in channels])
    PSD_AET_BATCHWISE= PSD_AET * xp.ones((batch_size, len(channels), len(PSD_AET[0])))
    #Draw samples from multivariate Gaussian
    variance_noise_f= 0.25* N_padded*dt*PSD_AET_BATCHWISE#N*PSD_AET_BATCHWISE/(4*dt)#should we use N_padded rather than N?
    noise_f = xp.random.normal(0,xp.sqrt(variance_noise_f)) + 1j*xp.random.normal(0,xp.sqrt(variance_noise_f))
    #Transforming the frequency domain signal into the time domain
    return xp.fft.irfft(noise_f)[:,:,:N]
def init_TDI(waveform_generator, dim, delta_t, tdi_chan="AE", orbit_obj=EqualArmlengthOrbits, use_gpu=True):
    #set T such that output length =>dim
    T= (dim*delta_t/YRSID_SI)+0.005#A tiny bit extra on T
    t0 = 20000.0   # How many samples to remove from start and end of simulations.
    order = 25
    # Specify whether we are using 1st gen. or 2nd gen. TDI
    tdi_gen = "2nd generation"#'1st generation'#
    index_lambda = 8
    index_beta = 7
    tdi_kwargs_esa = dict(
        order=order, tdi=tdi_gen, tdi_chan=tdi_chan,
        ) 
    orbits= orbit_obj(use_gpu=use_gpu)
    # return the response wrapper
    return ResponseWrapper(waveform_generator,T, delta_t,
                            index_lambda,index_beta,t0=t0,
                            flip_hx = True, use_gpu = use_gpu, is_ecliptic_latitude=False,
                            remove_garbage = True, orbits= orbits, **tdi_kwargs_esa)
# def init_TDI(dim, dt, TDI_channels="AET", use_gpu=True):#['TDIA','TDIE','TDIT']
#         T= (dim*dt/YRSID_SI)+0.005#A tiny bit extra on T to ensure output length =>dim
#         # order of the langrangian interpolation
#         t0 = 20000.0   # How many samples to remove from start and end of simulations
#         order = 25
#         orbit_file_esa = "/../../../../fred/oz303/aboumerd/software/lisa-on-gpu/orbit_files/esa-trailing-orbits.h5"
#         orbit_kwargs_esa = dict(orbit_file=orbit_file_esa) # these are the orbit files that you will have cloned if you are using Michaels code.
#         # you do not need to generate them yourself. They’re already generated. 

#         # 1st or 2nd or custom (see docs for custom)
#         tdi_gen = "2nd generation"
#         tdi_kwargs_esa = dict(
#             orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan=TDI_channels)

#         #Specify the indices of the sky coordinates in the array of parameters
#         index_lambda = 7 # Index of polar angle
#         index_beta = 8   # Index of phi angle

#         #Kwargs for the waveform generator
#         waveform_kwargs={"sum_kwargs":{"pad_output":True}}

#         #Initialise the waveform generator
#         generic_class_waveform_0PA_ecc = GenerateEMRIWaveform("FastSchwarzschildEccentricFlux", use_gpu = use_gpu, **waveform_kwargs)
#         #Then initialise the response wrapper
#         return ResponseWrapper(generic_class_waveform_0PA_ecc, T, dt,
#                                         index_lambda, index_beta, t0=t0,
#                                         flip_hx = True, use_gpu = use_gpu, is_ecliptic_latitude=False,
#                                         remove_garbage = True,  **tdi_kwargs_esa)


def generate_TDI_EMRI(EMRI_params, TDI_wrapper):
    '''
    Generate ONE EMRI using the initialised TDI/response wrapper
    from a set of parameters.
    '''
    return TDI_wrapper(*EMRI_params)
def get_TDI_noise(batch_size, dt, n_chans, len_arr, return_cupy=True):
    '''
    Generate ONE batch of TDI LISA noise. Not for overlaying on GW events
    since this is already whitened! More useful for tests involving pure noise.
    Output shape: (batch_size, dim, n_channels)
    '''
    if return_cupy==True:
        xp=cp
    else:
        xp=np
    batch_TDI_noise= noise_td_AET_BATCHWISE(len_arr, dt, batch_size, channels=["AE","AE"][:n_chans+1], return_cupy=return_cupy)
    batch_TDI_noise= noise_whiten_AET_BATCHWISE(batch_TDI_noise, dt, batch_size, channels=["AE","AE"][:n_chans+1])
    return batch_TDI_noise
def noise_whiten_AET(noisy_signal_td_AET, dt, PSD, window=None):#, channels=["AE","AE","T"]
    '''This is vectorised for the AET channels.        
        May be quicker or more convenient if we use PyTorch's FFT and windowing. Worth testing
        '''
    #Check whether to use cupy or numpy
    xp = cp.get_array_module(noisy_signal_td_AET)
    #Get signal length
    signal_length= len(noisy_signal_td_AET[0])
    #FFT the windowed TD signal; obtain freq bins
    noisy_signal_fd_AET = FFT_AET(noisy_signal_td_AET, zero_pad_data=True, window=window, fft_out=None)
    # window= xp.asarray(tukey(signal_length, alpha=0))# alpha=0,1/8
    # padded_noisy_signal_td_AET= xp.asarray([zero_pad(window*noisy_signal_td) for noisy_signal_td in noisy_signal_td_AET])
    # noisy_signal_fd_AET= xp.fft.rfft(padded_noisy_signal_td_AET)
    #Get padded td signal length
    # padded_signal_length= len(padded_noisy_signal_td_AET[0])
    # freq = xp.fft.rfftfreq(padded_signal_length, dt)
    # freq[0]=freq[1]#To avoid NaN in PSD[0]
    #Divide FD signal by ASD of noise
    # PSD_AET= xp.asarray([get_sensitivity(freq, sens_fn=A1TDISens, return_type="PSD") for channel in channels])
    scaling_factor= np.sqrt(2.0/(PSD*dt))
    whitened_signal_fd_AET= scaling_factor*noisy_signal_fd_AET
    #IRFFT and truncate the junk at the end
    whitened_signal_td_AET= xp.fft.irfft(whitened_signal_fd_AET, n=signal_length)#[:,:signal_length]#padded_signal_length
    #IFFTing back into the time domain
    return whitened_signal_td_AET
def noise_whiten_AET_BATCHWISE(noisy_signal_td_AET, dt, batch_size, channels=["AE","AE","T"]):
    '''
    Noise-whiten batchwise across the AET channels.
    It may also be quicker if we use PyTorch's FFT and windowing. Worth testing
    '''
    #Check whether to use cupy or numpy
    xp = cp.get_array_module(noisy_signal_td_AET)
    #FFT the windowed TD signal; obtain freq bins
    signal_length= noisy_signal_td_AET.shape[-1]
    window= xp.asarray(tukey(signal_length, alpha=0))#*xp.ones((batch_size, len(channels), signal_length))
    padded_noisy_signal_td_AET= zero_pad_BATCHWISE(window*noisy_signal_td_AET)#xp.asarray([zero_pad_BATCHWISE(window*noisy_signal_td) for noisy_signal_td in noisy_signal_td_AET])
    noisy_signal_fd_AET= xp.fft.rfft(padded_noisy_signal_td_AET)
    signal_length_padded= padded_noisy_signal_td_AET.shape[-1]
    freq = xp.fft.rfftfreq(signal_length_padded, dt)
    freq[0]=freq[1]#To avoid NaN in PSD[0]
    #Divide FD signal by ASD of noise
    PSD_AET= xp.asarray([get_sensitivity(freq, sens_fn=A1TDISens, return_type="PSD") for channel in channels])
    PSD_AET_BATCHWISE= PSD_AET# * xp.ones((batch_size, len(channels), len(PSD_AET[0])))
    scaling_factor= np.sqrt(2.0/(PSD_AET_BATCHWISE*dt))#((PSD_AET_BATCHWISE)/(2*dt))**-0.5#len(noisy_signal_td)
    whitened_signal_fd_AET= scaling_factor*noisy_signal_fd_AET
    #IFFTing back into the time domain; truncate the zero-padding
    whitened_signal_td_AET= xp.fft.irfft(whitened_signal_fd_AET, n=signal_length_padded)[:,:,:signal_length]
    return whitened_signal_td_AET
###
#other functions used after training to evaluate model accuracy and generalisability
###

def FFT_AET(signal_t_AET, zero_pad_data=True, window=None, fft_out=None):#window_data=True
    ''' Does a zero-padded windowed FFT.'''
    xp= cp.get_array_module(signal_t_AET)
    #Convert input to np array
    #signal_t_AET= xp.array(signal_t_AET)
    if window is not None:
        #window = xp.array(tukey(signal_t_AET.shape[-1], 0.05))
        signal_t_AET *= window
    if zero_pad_data == True:
        signal_t_AET = zero_pad(signal_t_AET)
    # Compute signal in frequency domain
    if fft_out is not None:
        fft_out[...] = xp.fft.rfft(signal_t_AET)
    else:
        fft_out = xp.fft.rfft(signal_t_AET)
    return fft_out
def inner_prod(sig1_t, sig2_t, N_t, delta_t, PSD, zero_pad_data=False, window=None):#, use_gpu=True
    """ This is only valid if:
        1. signals are same length
        2. signals have same no. of channels
    """
    xp= cp.get_array_module(sig1_t, sig2_t, PSD)
    # if use_gpu:#Fine to keep this; these variables are local
    #     xp=cp
    # else:
    #     xp=np
    
    #FFT the two signals
    sig1_f= FFT_AET(sig1_t, zero_pad_data=zero_pad_data, window=window, fft_out=None)#xp.fft.rfft(sig1_t)
    sig2_f_conj= FFT_AET(sig2_t, zero_pad_data=zero_pad_data, window=window, fft_out=None).conj()#xp.fft.rfft(sig2_t).conj()
    # # Get freq. bins
    # freq= xp.fft.fftfreq(N_t, delta_t)
    # freq[0] = freq[1]   # To "retain" the zeroth frequency
    # #Calculate the PSD
    # PSD_AET= xp.asarray([get_sensitivity(freq, sens_fn=A1TDISens, return_type="PSD") for channel in sig1_t.shape[0]])
    #Calculate the prefactor
    prefac = 4*delta_t / N_t
    #Calculate the output inn. prod.
    out= prefac* xp.real(xp.sum((sig1_f*sig2_f_conj)/PSD, axis=1))
    return out
def inner_prod_AET(sig1_t, sig2_t, delta_t, PSD):#, use_gpu=True
    """ Vectorised inner product for input shape (no. chans, length timeseries)
    This is only valid if:
        1. signals are same length
        2. signal has length 2**x
        3. signals have same no. of channels
    """
    xp= cp.get_array_module(sig1_t, sig2_t, PSD)
    # if use_gpu:#Fine to keep this; these variables are local
    #     xp=cp
    # else:
    #     xp=np
    #check the signals are the same length
    assert sig1_t.shape == sig2_t.shape
    N_t= sig1_t.shape[1]
    #FFT the two signals
    sig1_f= xp.fft.rfft(sig1_t)
    sig2_f_conj= xp.fft.rfft(sig2_t).conj()
    # #Get freq. bins
    # freq= xp.fft.rfftfreq(N_t, delta_t)
    # freq[0] = freq[1]   # To "retain" the zeroth frequency
    #Calculate the prefactor of inner prod.
    prefac = 4*delta_t / N_t
    #Calculate the output inn. prod.
    out= prefac* xp.real(xp.sum((sig1_f*sig2_f_conj)/PSD, axis=1))
    return out
def inner_prod_AET_batchwise(sig1_t_AET, sig2_t_AET, delta_t, PSD, use_gpu=True):#N_t,
    """ Vectorised inner product for input shape (batch size, no. chans, length timeseries)
    This is only valid if:
        1. signals are same length
        2. signal has length 2**x
        3. signals have same no. of channels
    """
    if use_gpu:#Fine to keep this; these variables are local
        xp=cp
    else:
        xp=np
    N_t= sig1_t_AET.shape[2]#len(sig1_t)
    #FFT the two signals
    freq= xp.fft.rfftfreq(N_t, delta_t)
    freq[0] = freq[1]   # To "retain" the zeroth frequency
    sig1_f= xp.fft.rfft(sig1_t_AET)
    sig2_f_conj= xp.fft.rfft(sig2_t_AET).conj()
    #Calculate the PSD
    PSD_AET= xp.asarray([get_sensitivity(freq, sens_fn=A1TDISens, return_type="PSD") for channel in range(sig1_t_AET.shape[1])])
    #PSD_AET= get_sensitivity(freq, sens_fn=A1TDISens, return_type="PSD")
    #Calculate the prefactor
    prefac = 4*delta_t / N_t
    #Calculate the output inn. prod.
    out= prefac* xp.real(xp.sum((sig1_f*sig2_f_conj)/PSD_AET, axis=2))
    return out
    '''Overlap across AET is:
    Overlap_AET= sqrt[(1/n_chans) * sum_across_chans(Overlap_chan^2)]
    '''
def overlap_AET(sig1_t_AET, sig2_t_AET, delta_t, PSD, use_gpu=True):
    """ Network overlap for input shape (no. chans, length timeseries)
    Formula taken from p9 of https://arxiv.org/pdf/2310.08927
    This is only valid if:
        1. signals are same length
        2. signal has length 2**x
        3. signals have same no. of channels
    """
    if use_gpu:#Fine to keep this; these variables are local
        xp=cp
    else:
        xp=np
    inner_prod_h1_h2= inner_prod_AET(sig1_t_AET, sig2_t_AET, delta_t, None, use_gpu=use_gpu)
    inner_prod_h1_h1= inner_prod_AET(sig1_t_AET, sig1_t_AET, delta_t, None, use_gpu=use_gpu)
    inner_prod_h2_h2= inner_prod_AET(sig2_t_AET, sig2_t_AET, delta_t, None, use_gpu=use_gpu)
    overlap= (inner_prod_h1_h2)/xp.sqrt(inner_prod_h1_h1* inner_prod_h2_h2)
    return xp.sqrt(xp.sum(overlap**2)/sig1_t_AET.shape[0])
def overlap_AET_batchwise(sig1_t_AET, sig2_t_AET, delta_t, PSD, use_gpu=True):
    """ Network overlap for input shape (batch size, no. chans, length timeseries)
    This is only valid if:
        1. signals are same length
        2. signal has length 2**x
        3. signals have same no. of channels
    """
    if use_gpu:#Fine to keep this; these variables are local
        xp=cp
    else:
        xp=np
    inner_prod_h1_h2= inner_prod_AET_batchwise(sig1_t_AET, sig2_t_AET, delta_t, None, use_gpu=use_gpu)
    inner_prod_h1_h1= inner_prod_AET_batchwise(sig1_t_AET, sig1_t_AET, delta_t, None, use_gpu=use_gpu)
    inner_prod_h2_h2= inner_prod_AET_batchwise(sig2_t_AET, sig2_t_AET, delta_t, None, use_gpu=use_gpu)
    overlap= (inner_prod_h1_h2)/xp.sqrt(inner_prod_h1_h1* inner_prod_h2_h2)
    return xp.sqrt(xp.sum(overlap**2, axis=1)/sig1_t_AET.shape[1])
def SNR_opt_AET(template, delta_t, PSD, zero_pad_data=True, window=None, return_each_channel=False):
    '''
    Computes the OPTIMAL SNR!
    waveform: list of form [A_chan, E_chan, T_chan]
    consider renaming this function to SNR_opt_AET
    '''
    xp= cp.get_array_module(template)
    #Convert tuple to np array
    template= xp.array(template)
    #Get the correct array length for the inner product
    if zero_pad_data == True:
        N_t = zero_pad_length(template)
    else:
        N_t= template.shape[-1]
    #Get freq-domain template
    # template_f= FFT_AET(template, zero_pad_data=zero_pad_data, window=window)
    # Compute squared optimal SNR
    SNR2_AET = inner_prod(template, template, N_t, delta_t, PSD, zero_pad_data=zero_pad_data, window=window)
    if return_each_channel == True:
        SNR = SNR2_AET**0.5
    else:
        SNR = xp.sqrt(xp.sum(SNR2_AET))
    return SNR
def get_target_luminosity_dist(target_SNR, previous_SNR, previous_luminosity_dist):
    '''
    Using some reference SNR and luminosity distance, calculate d_L corresponding to some target SNR.
    
    :param target_SNR: The SNR you want
    :param previous_SNR: The reference SNR
    :param previous_luminosity_dist: The reference luminosity distance
    '''
    return previous_luminosity_dist*previous_SNR/target_SNR
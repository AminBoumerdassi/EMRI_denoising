import torch
import torch.fft as torch_fft


def inner_prod_AET_batchwise(sig1_t_AET, sig2_t_AET, delta_t, PSD):
    """ Inner product for TENSORS of input shape (batch size, no. chans, length timeseries)
    This is only valid if:
        1. signals are same length
        2. signal has length 2**x
        3. signals have same no. of channels
    """
    N_t= sig1_t_AET.shape[2]#len(sig1_t)
    #FFT the two signals
    freq= torch_fft.rfftfreq(N_t, delta_t)
    freq[0] = freq[1]   # To "retain" the zeroth frequency
    sig1_f= torch_fft.rfft(sig1_t_AET)
    sig2_f_conj= torch_fft.rfft(sig2_t_AET).conj()
    #Calculate the prefactor
    prefac = 4*delta_t / N_t
    #Calculate the output inn. prod.
    if PSD is None:
        '''This assumes the data is already whitened'''
        out = prefac* torch.real(torch.sum((sig1_f*sig2_f_conj), axis=2))
    else:        
        out= prefac* torch.real(torch.sum((sig1_f*sig2_f_conj)/PSD, axis=2))
    return out
    '''Overlap across AET is:
    Overlap_AET= sqrt[(1/n_chans) * sum_across_chans(Overlap_chan^2)]
    '''

def overlap_AET_batchwise(sig1_t_AET, sig2_t_AET, delta_t, PSD):
    """ Network overlap for input shape (batch size, no. chans, length timeseries)
    This is only valid if:
        1. signals are same length
        2. signal has length 2**x
        3. signals have same no. of channels
    """
    inner_prod_h1_h2= inner_prod_AET_batchwise(sig1_t_AET, sig2_t_AET, delta_t, PSD)
    inner_prod_h1_h1= inner_prod_AET_batchwise(sig1_t_AET, sig1_t_AET, delta_t, PSD)
    inner_prod_h2_h2= inner_prod_AET_batchwise(sig2_t_AET, sig2_t_AET, delta_t, PSD)
    overlap= (inner_prod_h1_h2)/torch.sqrt(inner_prod_h1_h1* inner_prod_h2_h2)
    return torch.sqrt(torch.sum(overlap**2, axis=1)/sig1_t_AET.shape[1])
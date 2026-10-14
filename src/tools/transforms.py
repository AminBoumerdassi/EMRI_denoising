import numpy as np
import cupy as cp
import torch

def asinh_transform(input, a=1/1e-13, c=0., forward=True):
    '''Applies an asinh transform which has a loglike behaviour for sufficiently large x. Handles +ve and -ve inputs!
    a is the scale factor, c is the offset, inverse transform is set by forward=False'''
    if isinstance(input, np.ndarray):
        xp = np
    elif isinstance(input, cp.ndarray):
        xp = cp
    elif isinstance(input, torch.Tensor):
        xp = torch
    if forward:
        out = xp.asinh(a*input + c)
    else:
        out = (xp.sinh(input) - c ) / a
    return out

def signed_log_transform(input, forward=True):
    '''Applies a signed log transform.'''
    if isinstance(input, np.ndarray):
        xp = np
    elif isinstance(input, cp.ndarray):
        xp = cp
    elif isinstance(input, torch.Tensor):
        xp = torch
    '''We may need a small epsilon term for stability in the log'''
    eps=1e-12
    if forward:
        out = xp.sign(input) * xp.log(xp.abs(input) + eps)
    else:
        out = xp.sign(input) * xp.exp(xp.abs(input) + eps)
    return out


def normalise(input, scale_factor=1.0, forward=True):
    if forward:
        out = input/scale_factor
    else:
        out = scale_factor*input
    return out

def power_law_compression(input, power_factor=0.3, forward=True):
    if isinstance(input, np.ndarray):
        xp = np
    elif isinstance(input, cp.ndarray):
        xp = cp
    elif isinstance(input, torch.Tensor):
        xp = torch
    if forward:
        out = xp.sign(input) * xp.abs(input)**(power_factor)
    else:
        out = xp.sign(input) * xp.abs(input)**(1/power_factor)
    return out

def preprocess_signal(input, asinh_scale=1 / 1e-13, asinh_shift=0.0, max_absolute=None, inverse=False):
    '''
    Preprocesses input signals by first doing an asinh transform
    and scaling by the maximum absolute.
    max_abs_tensor= self.xp.array([30.990543, 30.842495])[:,None,None]
    '''
    if inverse == False:
        #Preprocess forwards
        if max_absolute is None:
            out = asinh_transform(input, a=asinh_scale, c=asinh_shift, forward=True)#signed_log_transform(input, forward=True)#
        else:
            out = asinh_transform(input, a=asinh_scale, c=asinh_shift, forward=True)#signed_log_transform(input, forward=True)#
            out = normalise(out, scale_factor=max_absolute, forward=True)
    else:
        #Preprocess backwards
        if max_absolute is None:
            out = asinh_transform(input, a=asinh_scale, c=asinh_shift, forward=False)#signed_log_transform(input, forward=False)#
        else:
            out = normalise(input, scale_factor=max_absolute, forward=False)
            out = asinh_transform(out, a=asinh_scale, c=asinh_shift, forward=False)#signed_log_transform(out, forward=False)#       
    return out

def preprocess_mask(input, power_factor=0.3, max_abs=None, scaling_factor=1.0, inverse=False):
    '''
    Preprocesses target masks by a reversible power law compression
    '''
    forward = False if inverse else True
    if inverse:
        out = normalise(input, scale_factor=max_abs, forward=forward)
        out = asinh_transform(out, a=scaling_factor, forward=forward)#signed_log_transform(out, forward=forward)#power_law_compression(input, power_factor=power_factor, forward=forward)
    else:
        out = asinh_transform(input, a=scaling_factor, forward=forward)#signed_log_transform(input, forward=forward)
        out = normalise(out, scale_factor=max_abs, forward=forward)
    # if inverse != False:
    #     out = power_law_compression(input, power_factor=power_factor, forward=True)
    # else:
    #     out = power_law_compression(input, power_factor=power_factor, forward=False)
    return out
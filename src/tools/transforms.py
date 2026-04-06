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

def normalise(input, scale_factor=1.0, forward=True):
    if forward:
        out = input/scale_factor
    else:
        out = scale_factor*input
    return out
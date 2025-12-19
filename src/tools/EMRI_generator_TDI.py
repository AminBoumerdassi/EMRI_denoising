'''
This is a custom Pytorch dataset for generating time-domain EMRIs from a given set of parameters.
It uses sets of EMRI parameters to generate and store time-domain EMRIs only for as long as is needed in a particular batch. 
'''
#---------------------------------------------------------------------------------------
#Adapted from: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
#---------------------------------------------------------------------------------------

import numpy as np
import cupy as cp
import copy
import torch

from .EMRI_analysis_tools import *

#FEW imports
import sys
import os
from numpy.random import default_rng
from few.trajectory.inspiral import EMRIInspiral
from few.waveform import Pn5AAKWaveform, GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux
from few.utils.geodesic import get_separatrix
from few.utils.utility import get_p_at_t
from few.utils.constants import YRSID_SI

#LISA tools imports
from lisatools.sensitivity import get_sensitivity, A1TDISens, T1TDISens
from lisatools.detector import EqualArmlengthOrbits, ESAOrbits
from lisatools.utils.constants import lisaLT

#fast lisa response imports
from fastlisaresponse import ResponseWrapper

#oise whitening/noise generation imports
from scipy.signal.windows import tukey


class EMRIGeneratorTDI(torch.utils.data.Dataset):
    'Generates data for PyTorch'
    def __init__(self, EMRI_params_and_SNRs, waveform_model="FastSchwarzschildEccentricFlux" ,dim=2**21, dt=10.,  TDI_channels="AE", target_SNR_range=[70,80],
                seed=2023, add_noise=True, use_gpu = True):
        'Initialization'
        self.EMRI_params_and_SNRs= copy.copy(EMRI_params_and_SNRs)#shape: (no. EMRIs, no. parameters + SNR)
        self.EMRI_params_set_size= self.EMRI_params_and_SNRs.shape[0]
        self.reference_SNRs = copy.copy(EMRI_params_and_SNRs[:,-1])
        self.reference_luminosity_dists = copy.copy(EMRI_params_and_SNRs[:,6])
        self.dim = dim
        self.dt = dt
        self.TDI_channels=TDI_channels
        self.T= (dim*dt/YRSID_SI)+0.005#A tiny bit extra on T to ensure output length =>dim
        self.channels_dict= {"AET":["AE","AE","T"], "AE":["AE","AE"]}        
        #For use in the noise generation and whitening functions
        self.n_channels = len(TDI_channels)
        self.add_noise= add_noise
        self.seed= seed
        self.use_gpu= use_gpu
        #Gpu check
        if self.use_gpu == True:
            self.xp = cp
            self.backend_type= "cuda12x"
            self.device="cuda"
        else:
            self.xp =np
            self.backend_type= "cpu"
            self.device="cpu"
        #initialise RNG for noise generation with a fixed seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        # self.random.seed(self.seed)
        #Initialise the waveform generator
        inspiral_kwargs = {}#'flux_output_convention':'pex'
        sum_kwargs = {"pad_output": True}
        amplitude_kwargs = {}
        waveform_generator = GenerateEMRIWaveform(waveform_model,
                                           force_backend=self.backend_type,
                                             sum_kwargs=sum_kwargs,
                                             inspiral_kwargs=inspiral_kwargs,
                                             amplitude_kwargs=amplitude_kwargs)
        #Initialise the TDI wrapper
        self.TDI_wrapper= init_TDI(waveform_generator, self.dim, self.dt, tdi_chan=TDI_channels, orbit_obj=EqualArmlengthOrbits, use_gpu=use_gpu)
        # Construct the PSD
        self.PSD_AE = noise_PSD_AE(dim, dt, TDI = 'TDI2', include_foreground=True, model='scirdv1', xp=self.xp)
        #Tune the luminosity distances to ensure the SNRs are in the target range
        self.retune_SNRs(low=target_SNR_range[0], high=target_SNR_range[1])
    def __len__(self):
        'Denotes the total number of samples'
        #This could be calculated in terms of batch size and batches per epochs i.e. BS*B_per_epoch
        return 1024#128
    def __getitem__(self, index):
        'Generates one sample of data'
        X, y = self.data_generation(index)
        #Do a batch-wise noise-gen
        #Do a batch-wise summation of waveform and noise
        #Do a batch-wise noise-whitening
        #Do a batch-wise conversion of xp arr to torch tensor
        return X, y            
    def data_generation(self, index):
        'Generate a single noise-whitened TDI EMRI.'
        #Generate the list of waveforms, convert to array, then truncate to the correct length
        waveform= self.xp.asarray(generate_TDI_EMRI(self.EMRI_params_and_SNRs[index,:-1], self.TDI_wrapper))[:,:self.dim]
        #Then preprocess with noise and whitening
        if self.add_noise==True:
            noise_AET= noise_td_AET(self.dim, self.dt, self.PSD_AE, return_cupy=self.use_gpu)
            noisy_signal_AET= waveform+noise_AET
            '''We could try other kinds of noise such as data gaps and glitches.'''
        else:
            noisy_signal_AET= waveform
        #Whiten X and y
        X= noise_whiten_AET(noisy_signal_AET, self.dt, self.PSD_AE, window=None)#noisy_signal_AET#
        y= noise_whiten_AET(waveform, self.dt, self.PSD_AE, window=None)#waveform#
        #Convert X from xp arrays to PyTorch tensors
        X= torch.as_tensor(X, device=self.device).float()
        y= torch.as_tensor(y, device=self.device).float()
        return X, y 
    def declare_generator_params(self):
        #Declare generator parameters
        print("#################################")
        print("####DATASET PARAMETERS####")
        print("#Dataset size: ", self.EMRI_params_set_size)
        print("#Time in years:", self.T)
        print("#n_channels: ", self.n_channels)
        print("#dt in seconds: ",self.dt)
        print("#Length of timeseries:", self.dim)
        print("Noise background: ", self.add_noise)
        print("#################################")
    def retune_SNRs(self, low=70, high=80):
        #Calculate the luminosity distances that make the SNRs uniformly distributed in the target range
        target_SNRs = np.random.uniform(low=low, high=high, size=self.EMRI_params_set_size)
        target_luminosity_dist = get_target_luminosity_dist(target_SNRs, self.reference_SNRs, self.reference_luminosity_dists)
        #Reset the luminosity distances in the parameters
        self.EMRI_params_and_SNRs[:,6] = target_luminosity_dist

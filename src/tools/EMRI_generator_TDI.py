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
import torch.utils.dlpack as dlpack

import pytorch_pfn_extras as ppe

import glob
from h5py import File

from .EMRI_analysis_tools import *

from .transforms import *

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
import scipy.signal as sp_signal
import cupyx.scipy.signal as cp_signal

#pywavelet import
import pywavelet
# from pywavelet.backend import *
# from pywavelet import set_backend
# from pywavelet.types import FrequencySeries, Wavelet, TimeSeries

#Disable pywavelet logging of warnings
import logging
logging.getLogger("pywavelet").setLevel(logging.ERROR)



class EMRIGeneratorTDI(torch.utils.data.Dataset):
    'Generates data for PyTorch'
    def __init__(self, EMRI_params_and_SNRs, waveform_model="FastSchwarzschildEccentricFlux" ,
                 dim=2**21, dt=10.,  TDI_channels="AE", target_SNR_range=[70,80],
                seed=2023, add_noise=False, add_glitches=False, use_gpu = True, random_windows=False,
                wavelet_transform=False, Nt=None, Nf=None, nx=None, mult=None):
        'Initialization'
        self.EMRI_params_and_SNRs= copy.copy(EMRI_params_and_SNRs)#shape: (no. EMRIs, no. parameters + SNR)
        self.EMRI_params_set_size= self.EMRI_params_and_SNRs.shape[0]
        self.reference_SNRs = copy.copy(EMRI_params_and_SNRs[:,-1])
        self.reference_luminosity_dists = copy.copy(EMRI_params_and_SNRs[:,6])
        self.dim = dim
        self.dt = dt
        self.TDI_channels=TDI_channels
        self.channels_dict= {"AET":["AE","AE","T"], "AE":["AE","AE"]}
        self.random_windows = random_windows
        #For use in the noise generation and whitening functions
        self.n_channels = len(TDI_channels)
        self.add_noise= add_noise
        self.seed= seed
        self.use_gpu= use_gpu
        #For use in the glitch generation
        self.add_glitches = add_glitches
        self.glitch_scale_factor = 1.0
        #Pywavelet kwargs
        self.wavelet_transform=wavelet_transform
        self.Nt=Nt
        self.Nf=Nf
        self.nx=nx
        self.mult=mult
        #Gpu check
        if self.use_gpu == True:
            self.xp = cp
            self.xp_signal = cp_signal
            self.backend_type= "cuda12x"
            self.device="cuda"
            #Ensure cupy mem pooling enabled
            self.xp.get_default_memory_pool()
            #Make cupy use the same stream as torch
            torch_stream = torch.cuda.current_stream().cuda_stream
            self.xp.cuda.ExternalStream(torch_stream)
            #Also share memory pool between torch and cp
            '''NOTE: this makes errors messages extremely verbose. Temporarily disable whenever debugging is needed'''
            ppe.cuda.use_torch_mempool_in_cupy()
        else:
            self.xp =np
            self.xp_signal = sp_signal
            self.backend_type= "cpu"
            self.device="cpu"
        #initialise RNG for noise generation with a fixed seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        # self.random.seed(self.seed)
        #Initialise the waveform generator
        inspiral_kwargs = {}
        sum_kwargs = {"pad_output": True}
        amplitude_kwargs = {}
        waveform_generator = GenerateEMRIWaveform(waveform_model,
                                           force_backend=self.backend_type,
                                             sum_kwargs=sum_kwargs,
                                             inspiral_kwargs=inspiral_kwargs,
                                             amplitude_kwargs=amplitude_kwargs)
        #Initialise the backend for pywavelet
        if wavelet_transform:
            pywavelet.set_backend("cupy", "float64") if use_gpu else pywavelet.set_backend("numpy", "float64")
            #Initialise the timeseries objects for X and y
            self.X_timeseries = pywavelet.types.TimeSeries(self.xp.zeros(dim), self.xp.arange(0, self.dim*self.dt, dt))
            self.y_timeseries = pywavelet.types.TimeSeries(self.xp.zeros(dim), self.xp.arange(0, self.dim*self.dt, dt))
            #Initialise an empty array for the wavelet transforms
            self.X_tf= self.xp.zeros((len(TDI_channels), self.Nf, self.Nt))
            self.y_tf= self.xp.zeros((len(TDI_channels), self.Nf, self.Nt))
        #Set the time window to 2 years if we want randomised windows of EMRIs
        if self.random_windows:
            self.T= 2.0
            long_dim = int(self.T*365*86400/dt)
            #Init TDI wrapper
            self.TDI_wrapper= init_TDI(waveform_generator, long_dim, self.dt, tdi_chan=TDI_channels, orbit_obj=EqualArmlengthOrbits, use_gpu=use_gpu)
        else:
            self.T= (dim*dt/YRSID_SI)+0.005#A tiny bit extra on T to ensure output length =>dim
            #Initialise the TDI wrapper
            self.TDI_wrapper= init_TDI(waveform_generator, self.dim, self.dt, tdi_chan=TDI_channels, orbit_obj=EqualArmlengthOrbits, use_gpu=use_gpu)
        # Construct the PSD
        self.PSD_AE = noise_PSD_AE(dim, dt, TDI = 'TDI2', include_foreground=True, model='scirdv1', xp=self.xp)
        #Tune the luminosity distances to ensure the SNRs are in the target range
        self.retune_SNRs(low=target_SNR_range[0], high=target_SNR_range[1])
        #Load a glitch background if needed, rescale, and resample
        if self.add_glitches:
            #Pick a particular glitch background
            self.background_idx=0
            #Load & resample it
            self.load_glitch_background(glitch_background_dir="/fred/oz303/aboumerd/EMRI_Glitches/data_files/glitch_bg_AET/max_glitch_SNR_inf/",
                                                            background_idx=self.background_idx)
            #Resample the cropped glitch background to match the 0.1Hz fs
            # self.glitches_td_AET = self.xp.array([self.xp_signal.resample_poly(self.glitches_td_AET[channel,:], 1,5, axis=-1) for channel in range(self.glitches_td_AET.shape[0])])
            #Initialise the rescaled glitches
            self.retune_glitches(glitch_scale_factor=1.0)
            # self.glitch_scale_factor = 1.0
            # self.rescaled_glitches_td_AET = self.glitch_scale_factor * self.glitches_td_AET
    def __len__(self):
        'Denotes the total number of samples in the dataset'
        return self.EMRI_params_set_size
    def __getitem__(self, index):
        'Generates one sample of data'
        X, y, mask = self.data_generation(index)
        #Do a batch-wise noise-gen
        #Do a batch-wise summation of waveform and noise
        #Do a batch-wise noise-whitening
        #Do a batch-wise conversion of xp arr to torch tensor
        return X, y, mask            
    def data_generation(self, index):
        'Generate a single noise-whitened TDI EMRI.'
        #Choose a random start point in the window if needed
        if self.random_windows:
            end_idx = self.xp.random.randint(self.dim, self.TDI_wrapper.response_model.num_pts)
            start_idx = end_idx - self.dim
            waveform= self.xp.asarray(generate_TDI_EMRI(self.EMRI_params_and_SNRs[index,:-1], self.TDI_wrapper))[:,start_idx:end_idx]
        else:
            #Generate the list of waveforms, convert to array, then truncate to the correct length
            waveform= self.xp.asarray(generate_TDI_EMRI(self.EMRI_params_and_SNRs[index,:-1], self.TDI_wrapper))[:,:self.dim]
        #Initialise the input signal array
        signal_AET = self.xp.zeros_like(waveform)#self.xp.copy(waveform)
        signal_AET += waveform
        #Add noise and glitches if desired
        if self.add_noise:
            signal_AET += noise_td_AET(self.dim, self.dt, self.PSD_AE, return_cupy=self.use_gpu)
        if self.add_glitches:
            '''Much too slow to load a new glitch background for every training sample.
            Stick to loading once per epoch.'''
            # #Specify the directory of glitch backgrounds
            # glitch_background_dir = "/fred/oz303/aboumerd/EMRI_Glitches/data_files/glitch_bg_AET/max_glitch_SNR_inf/"
            # #Glob the glitch backgrounds
            # glitch_background_files = glob.glob(glitch_background_dir + "BG_*.h5")
            # #Randomly select a file from the globbed list
            # rand_idx = np.random.randint(0, len(glitch_background_files)-1)
            # #Randomly choose a file
            # glitch_background_file = glitch_background_files[rand_idx]
            # #Load file
            # with File(glitch_background_file, 'r') as f:
            #     '''
            #     Note: the cupy implementation of resample_poly is broken
            #     for multidimensional arrays so use it channel by channel.
            #     '''
            #     long_glitches_td_AET = self.xp.array([f["A"][()], f["E"][()]])#
            #     glitch_dt= f["dt"][()]#Original fs of 0.5Hz
            # glitch_target_len= int(self.dt*self.dim/glitch_dt)
            #Create a randomised window of the 2y glitch background
            start_idx = np.random.randint(0, self.glitches_td_AET.shape[-1] - signal_AET.shape[-1] - 1)
            end_idx = start_idx + signal_AET.shape[-1]
            #Crop the background to match the length of the input signal
            cropped_glitches_td_AET = self.rescaled_glitches_td_AET[: , start_idx:end_idx]#signal_AET.shape[-1]
            # #Resample the cropped glitch background to match the 0.1Hz fs
            # glitches_td_AET = self.xp.array([self.xp_signal.resample_poly(cropped_glitches_td_AET[channel,:], 1,5, axis=-1) for channel in range(cropped_glitches_td_AET.shape[0])])
            #Randomly flip the glitch background by +-1
            cropped_glitches_td_AET *= self.xp.random.choice([1.0,-1.0], size=1)
            #Rescale the glitch backgrounds by some factor as a form of curriculum learning
            # self.glitches_td_AET *= self.glitch_scale_factor
            # #Convert to cupy array if needed
            # if self.use_gpu: glitches_td_AET = self.xp.asarray(glitches_td_AET)
            signal_AET += cropped_glitches_td_AET
        #Whiten X and y
        X_t= noise_whiten_AET(signal_AET, self.dt, self.PSD_AE, window=None)
        y_t= noise_whiten_AET(waveform, self.dt, self.PSD_AE, window=None)
        #Convert to time-freq domain if desired
        if self.wavelet_transform:
            '''Annoyingly, pywavelet doesn't do multichannel transforms. So we have to transform
            each channel separately.'''
            for chan in range(len(self.TDI_channels)):
                self.X_timeseries.data = X_t[chan]
                self.y_timeseries.data = y_t[chan]
                #Transform the timeseries to t-f
                #Note that the transforms produce real output that can be +ve or -ve and vary across orders of magnitude
                self.X_tf[chan] = pywavelet.transforms.from_time_to_wavelet(self.X_timeseries, Nf=self.Nf, Nt=self.Nt, nx=self.nx, mult=self.mult).data#self.X_timeseries.to_wavelet(Nf=self.Nf, Nt=self.Nt).data
                self.y_tf[chan] = pywavelet.transforms.from_time_to_wavelet(self.y_timeseries, Nf=self.Nf, Nt=self.Nt, nx=self.nx, mult=self.mult).data#self.y_timeseries.to_wavelet(Nf=self.Nf, Nt=self.Nt).data
            #Plug into X and y
            X= self.X_tf
            y= self.y_tf
        else:
            X= X_t
            y= y_t
        #Rescale X and y with an asinh transformation
        max_abs_tensor= self.xp.array([30.990543, 30.842495])[:,None,None]#torch.as_tensor([37.57473783, 36.71401857], device=device)[None,:,None,None]
        X = asinh_transform(X, a=1 / 1e-13, c=0.0, forward=True)
        X= normalise(X, scale_factor=max_abs_tensor, forward=True)
        y = asinh_transform(y, a=1 / 1e-13, c=0.0, forward=True)
        y= normalise(y, scale_factor=max_abs_tensor, forward=True)
        #Calculate the complex mask
        mask_X = y/X
        #Convert data to float32 precision if needed
        X= X.astype(self.xp.float32, copy=False)
        y= y.astype(self.xp.float32, copy=False)
        mask_X= mask_X.astype(self.xp.float32, copy=False)
        #Convert X and y from xp arrays to PyTorch tensors
        X= torch.from_dlpack(X)#torch.utils.dlpack.from_dlpack(X.toDlpack()) #torch.as_tensor(X, device=self.device).float()
        y= torch.from_dlpack(y)#torch.utils.dlpack.from_dlpack(y.toDlpack())#torch.as_tensor(y, device=self.device).float()
        mask_X = torch.from_dlpack(mask_X)#torch.utils.dlpack.from_dlpack(mask_X.toDlpack())#torch.as_tensor(mask_X, device=self.device).float()
        return X, y, mask_X
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
    def load_glitch_background(self, glitch_background_dir=None, background_idx=0):
        #Glob the glitch background files
        glitch_background_files = glob.glob(glitch_background_dir + "BG_*.h5")
        #Choose the specified one
        glitch_background_file = glitch_background_files[background_idx]
        #Load file
        with File(glitch_background_file, 'r') as f:
            '''
            Note: the cupy implementation of resample_poly is broken
            for multidimensional arrays so use it channel by channel.
            '''
            self.glitches_td_AET = self.xp.array([f["A"][()], f["E"][()]])#
            glitch_dt= f["dt"][()]#Original fs of 0.5Hz
        #Resample the glitch background
        self.glitches_td_AET = self.xp.array([self.xp_signal.resample_poly(self.glitches_td_AET[channel,:], 1,5, axis=-1) for channel in range(self.glitches_td_AET.shape[0])])
    def retune_glitches(self, glitch_scale_factor=1.0):
        '''Rescale the glitch backgrounds by multiplying by alpha'''
        #Change the scale factor
        self.glitch_scale_factor = glitch_scale_factor
        self.rescaled_glitches_td_AET = self.glitch_scale_factor * self.glitches_td_AET

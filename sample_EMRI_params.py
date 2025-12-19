'''
Took roughly 40 mins to run for 10k EMRIs.
'''
import numpy as np
import cupy as cp

#Set up a random number generator
# from numpy.random import default_rng
# rng = default_rng(seed=2024)

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

from src.tools.EMRI_analysis_tools import *
from src.tools.EMRI_param_sampler import *

use_gpu=True
seed=2024

#Gpu check
if use_gpu == True:
    xp = cp
    backend_type= "cuda12x"
    device="cuda"
    cp.random.seed(seed)
    np.random.seed(seed)
else:
    xp =np
    backend_type= "cpu"
    device="cpu"
    np.random.seed(seed)

#How many EMRIs do we want?
EMRI_count=10000

#Initialise waveform generator
waveform_model="FastSchwarzschildEccentricFlux"
dim=2**23
dt=10.
T= (dim*dt/YRSID_SI)+0.005
inspiral_kwargs = {}#'flux_output_convention':'pex'
sum_kwargs = {"pad_output": True}
amplitude_kwargs = {}
TDI_channels="AE"
waveform_generator = GenerateEMRIWaveform(waveform_model,
                                    force_backend=backend_type,
                                        sum_kwargs=sum_kwargs,
                                        inspiral_kwargs=inspiral_kwargs,
                                        amplitude_kwargs=amplitude_kwargs)

#Initialise the TDI wrapper
TDI_wrapper= init_TDI(waveform_generator, dim, dt, tdi_chan=TDI_channels, orbit_obj=EqualArmlengthOrbits, use_gpu=use_gpu)

# Construct the PSD
PSD_AE = noise_PSD_AE(dim, dt, TDI = 'TDI2', include_foreground=True, model='scirdv1', xp=xp)

#Sample EMRI params
EMRI_params= sample_EMRI_parameters(no_EMRIs=EMRI_count)

#Create array of SNRs to be calculated
SNR_arr= xp.zeros((EMRI_count,1))

#Create buffer for waveform generation
# waveform = xp.empty((len(TDI_channels), TDI_wrapper.response_model.num_pts))

#Iterate SNR calculation for each set of EMRI params
for i in range(EMRI_count):
    print(f"Waveform no. {i}/{EMRI_count}")
    #Generate waveform
    # xp.stack(generate_TDI_EMRI(EMRI_params[i,:], TDI_wrapper), out=waveform, axis=1)
    waveform= xp.asarray(generate_TDI_EMRI(EMRI_params[i,:], TDI_wrapper))[:,:dim]
    #Calculate and store SNR of a given EMRI
    SNR_arr[i]= SNR_opt_AET(waveform, dt, PSD_AE, zero_pad_data=True, window=None, return_each_channel=False)

#Convert the SNR to np if needed
if use_gpu:
    SNR_arr=SNR_arr.get()    

#Combine the params and SNRs, and save
EMRI_params_and_SNRs = np.concatenate((EMRI_params, SNR_arr), axis=1)

save_dir= "/fred/oz303/aboumerd/EMRI_denoising/training_data/"
fname= "EMRI_params.npy"
np.save(save_dir+fname, EMRI_params_and_SNRs)

'''
I suspect only a handful of the EMRIs will have SNRs <80, but we should check.
Load the SNR array up and we can delete any EMRI params corresponding to those bad EMRIs.
'''
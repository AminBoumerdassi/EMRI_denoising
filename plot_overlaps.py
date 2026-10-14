#numpy and matplotlib imports
import numpy as np
import matplotlib.pyplot as plt

#torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from ema_pytorch import EMA
from autoclip.torch import QuantileClip

#sklearn imports
from sklearn.model_selection import train_test_split

#Other imports
from few.utils.constants import YRSID_SI
import os
import copy
import random
import datetime

#pywavelet imports
import pywavelet
from pywavelet.types.wavelet_bins import compute_bins

#Disable pywavelet logging of warnings
import logging
logging.getLogger("pywavelet").setLevel(logging.ERROR)

#Set the default precisions, backends and device for torch
torch.set_default_dtype(torch.float32)###torch.float64
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#Set the default precisions/backends for pywavelet
pywavelet.set_backend("cupy", "float32") if use_cuda else pywavelet.set_backend("numpy", "float32")#"float64","float64"

# Data generator, training loop and net architecture imports
from src.tools.EMRI_generator_TDI import EMRIGeneratorTDI
from src.tools.test_and_train_loop import *
from src.tools.model_architecture import *
from src.tools.losses import *
from src.tools.transforms import *

#For debugging (ONLY turn on when needed)
torch.autograd.set_detect_anomaly(False)

# Using mixed precision
use_amp=True#False
scaler = torch.amp.GradScaler(device, enabled=use_amp)

# Reproducibility & determinism for pytorch, numpy and cupy
seed=2026
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # numpy random generator
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
g = torch.Generator()
g.manual_seed(seed)

# Initialise the model and move it to device
model= ConvAE_2D().to(device)#Dilated_ConvAE().to(device)#ConvAE().to(device)

# Make model in channels-last format for speedup
model = model.to(memory_format=torch.channels_last)

#Trick: compile the model
model = torch.compile(model)

checkpoint_path = "model_checkpoint.tar"#"/fred/oz303/aboumerd/EMRI_denoising/trained_nearly_model_with_CBAM_and_ASPP/model_checkpoint.tar"#"model_checkpoint.tar"#None#"model_checkpoint.tar"#None#"model_checkpoint.tar"#None#"model_checkpoint.tar"#None#"model_checkpoint.tar"#

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

#load model weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

#Setting generator parameters
len_seq= 2**20#23#2**22 gives around 1.3 years, 2**23 around 2.6 years
dt=10.#10
fs=1/dt
T= len_seq*dt/round(YRSID_SI)#Not actually input into the generator, it already calculates this and stores as an attribute
TDI_channels="AE"
n_channels=len(TDI_channels)

add_noise=False#True#False
add_glitches=True#True#True
random_sign_flips=True


training_target_SNR_range = [20, 80]
validation_target_SNR_range = [20, 80]
random_windows=False#True

#pywavelet kwargs
wavelet_transform = True#None
'''NOTE: 500 by 500 doesn't capture the full timeseries but for memory constraints,
we'll just have to settle on it rather than 1000 by 1000'''
Nt = 512#500#256#None
Nf = 512#500#256#None
nx = 8.0#None
mult = Nt//2#None

#Setting training hyperparameters
batch_size=32
epochs=2000#200#100
test_size=0.3

#Load dataset of EMRI parameters and apply test-train split
EMRI_params_dir="training_data/EMRI_params.npy"
EMRI_params_and_SNRs= np.load(EMRI_params_dir, allow_pickle=True)
train_params, val_params= train_test_split(EMRI_params_and_SNRs, test_size=test_size, random_state=seed)

#Initialise the EMRI generators for the training and val data
training_set= EMRIGeneratorTDI(train_params, 
                               dim=len_seq, 
                               dt=dt, 
                               TDI_channels=TDI_channels, 
                               add_noise=add_noise, 
                               add_glitches=add_glitches, 
                               seed=seed, 
                               target_SNR_range=training_target_SNR_range, 
                               use_gpu=use_cuda, 
                               random_windows=random_windows, 
                               wavelet_transform=wavelet_transform, 
                               Nt=Nt, 
                               Nf=Nf, 
                               nx=nx, 
                               mult=mult, 
                               random_sign_flips=random_sign_flips)

validation_set= EMRIGeneratorTDI(val_params, 
                                dim=len_seq, 
                                dt=dt, 
                                TDI_channels=TDI_channels, 
                                add_noise=add_noise, 
                                add_glitches=add_glitches, 
                                seed=seed, 
                                target_SNR_range=validation_target_SNR_range, 
                                use_gpu=use_cuda, 
                                random_windows=False, 
                                wavelet_transform=wavelet_transform, 
                                Nt=Nt, 
                                Nf=Nf, 
                                nx=nx, 
                                mult=mult, 
                                random_sign_flips=False)

#Get current states so we can reset the val set at each epoch
val_torch_rng_state = torch.get_rng_state()
val_cuda_rng_state = torch.cuda.get_rng_state_all()
val_numpy_rng_state = np.random.get_state()
val_python_rng_state = random.getstate()

#Set the current state for the EMRI generators
training_set.set_seed_state(val_torch_rng_state, 
                            val_cuda_rng_state, 
                            val_numpy_rng_state)

validation_set.set_seed_state(val_torch_rng_state, 
                            val_cuda_rng_state, 
                            val_numpy_rng_state)

#Apply Repeated Sampling of Random Subsets on training data to save time
samples_per_epoch = 1024
sampler = torch.utils.data.RandomSampler(training_set, 
                                         num_samples=samples_per_epoch, 
                                         replacement=False, 
                                         generator=g)

#Use a fixed subset of the validation to save time
no_val_samples= 1024
validation_subset= torch.utils.data.Subset(validation_set, 
                                           np.arange(no_val_samples).tolist())

#Initialise training dataloader
training_dataloader= torch.utils.data.DataLoader(training_set, 
                                                 batch_size=batch_size, 
                                                 sampler=sampler, 
                                                 shuffle=None, 
                                                 drop_last=True, 
                                                 generator=g)

#Initialise validation dataloader
'''Generator needs a fixed state with NO data augmentation'''
val_g = torch.Generator()
val_g.manual_seed(seed)
validation_dataloader= torch.utils.data.DataLoader(validation_subset, 
                                                   batch_size=batch_size, 
                                                   shuffle=True, 
                                                   drop_last=True, 
                                                   generator=val_g)

#Load example val. data without graphs or extra things we don't need
'''NOW USING FULL STRENGTH GLITCH BACKGROUNDS'''
glitch_background_dir= "/fred/oz303/aboumerd/EMRI_Glitches/data_files/glitch_bg_AET/max_glitch_SNR_inf/"
background_idx = 0

with torch.inference_mode():
    '''IMPORTANT: Using mitigated glitch backgrounds for the val. data!!!!'''
    validation_set.load_glitch_background(glitch_background_dir=glitch_background_dir,
                                                                background_idx=background_idx)
    X_EMRIs, y_true_EMRIs, residual, mask_EMRIs, mask_residual = next(iter(validation_dataloader))

#Restrict ourselves to the 1st sample in the batch
#And convert to channels last format
X_EMRIs = X_EMRIs[:1,:,:,:].to(memory_format=torch.channels_last)
y_true_EMRIs = y_true_EMRIs[:1,:,:,:].to(memory_format=torch.channels_last)
mask_EMRIs = mask_EMRIs[:1,:,:,:].to(memory_format=torch.channels_last)
mask_residual = mask_residual[:1,:,:,:].to(memory_format=torch.channels_last)
residual = residual[:1,:,:,:].to(memory_format=torch.channels_last)

def Overlap_tf_Metric(input, target, eps=1e-4):
    '''
    Return the unreduced overlap in a given batch of data. 
    This is ONLY valid when the input and target are already whitened.
    '''
    #Cast objects to float64 precision
    input_64 = input.double()
    target_64 = target.double()
    input_dot_target = torch.nansum(input_64*target_64, dim=(-2,-1))
    target_dot_target = torch.nansum(target_64*target_64, dim=(-2,-1))
    input_dot_input = torch.nansum(input_64*input_64, dim=(-2,-1))
    #Calculate denominator terms individually
    input_norm  = torch.sqrt(input_dot_input + eps)#torch.sqrt(torch.clamp(input_dot_input, min=self.eps))
    target_norm = torch.sqrt(target_dot_target + eps)#torch.sqrt(torch.clamp(target_dot_target, min=self.eps))
    #Calculate the match
    match = input_dot_target / (input_norm * target_norm)#input_dot_target * torch.rsqrt(target_dot_target * input_dot_input + self.eps)
    #match = torch.mean(match)
    return match.float()

#Define lists that we'll append the overlaps to
clean_overlaps_list = []
noisy_overlaps_list = []

#Train the model
no_iterations=100
for t in range(no_iterations):
    print(f"Iteration {t}")
    #Apply per-epoch data augmentations on validation set
    validation_set.retune_SNRs(low=validation_target_SNR_range[0], high=validation_target_SNR_range[1])
    X_EMRIs, y_true_EMRIs, residual, mask_EMRIs, mask_residual = next(iter(validation_dataloader))
    #Do val. forward pass in inference mode and mixed precision
    with torch.inference_mode():
        with torch.autocast(device_type=str(device), dtype=torch.bfloat16, enabled=use_amp):
            '''Not using ema model for testing'''
            pred_masks, _ = model(X_EMRIs)
        #Unpreprocess mask
        max_abs_tensor= 53.0#106.0#
        inverse=True
        asinh_scale= 1e23#1e46#
        max_abs= 80.0#45.0
        scaling_factor= 1e28#1e14
        #Backtransform input, target, and residual
        X_bt = preprocess_signal(X_EMRIs, 
                            asinh_scale=asinh_scale, 
                            # asinh_shift=asinh_shift, 
                            max_absolute=max_abs_tensor,
                            inverse = inverse
                            ).float()
        y_bt = preprocess_signal(y_true_EMRIs, 
                            asinh_scale=asinh_scale, 
                            # asinh_shift=asinh_shift, 
                            max_absolute=max_abs_tensor,
                            inverse = inverse
                            ).float()
        #Apply the denoising mask to the backtransformed data
        pred_denoised_bt = X_bt * preprocess_mask(pred_masks, 
                                    max_abs=max_abs,
                                    scaling_factor=scaling_factor,
                            #  power_factor=power_factor, 
                                inverse=inverse
                                ).float()
    #Calculate overlaps
    overlaps_clean = Overlap_tf_Metric(pred_denoised_bt, y_bt, eps=1e-15)
    overlaps_noisy = Overlap_tf_Metric(X_bt, y_bt, eps=1e-15)
    #Append ovelaps
    clean_overlaps_list += overlaps_clean.flatten().tolist()
    noisy_overlaps_list += overlaps_noisy.flatten().tolist()
    # clean_overlaps_list.append(overlaps_clean.flatten().tolist())
    # noisy_overlaps_list.append(overlaps_noisy.flatten().tolist())

#Save data
clean_overlaps = np.array(clean_overlaps_list)
noisy_overlaps = np.array(noisy_overlaps_list)

np.save("clean_overlaps.npy", clean_overlaps)
np.save("noisy_overlaps.npy", noisy_overlaps)

#Plot histograms of the noisy and clean overlaps
no_bins=100
plt.figure()
fig, axs = plt.subplots(2,1)
axs[0].hist(clean_overlaps, label="Denoised data", bins=no_bins)
axs[1].hist(noisy_overlaps, label="Noisy data", bins=no_bins)
# axs[0].set_xlabel("Overlap")
axs[0].set_ylabel("Counts")
axs[1].set_xlabel("Overlaps across A and E channels")
axs[1].set_ylabel("Counts")

plt.savefig("glitchy_EMRI_overlaps_fig.pdf")

#print("Done!")

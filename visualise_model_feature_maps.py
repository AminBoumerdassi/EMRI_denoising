#numpy and matplotlib imports
import numpy as np
import matplotlib.pyplot as plt

#torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torchinfo import summary

#sklearn imports
from sklearn.model_selection import train_test_split

#Other imports
from few.utils.constants import YRSID_SI
import os
import copy
import random

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
# from src.tools.test_and_train_loop import *
from src.tools.model_architecture import *
from src.tools.losses import *
from src.tools.transforms import *

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

checkpoint_path = "/fred/oz303/aboumerd/EMRI_denoising/trained_model_full_glitches/model_checkpoint.tar"

#Load checkpoint dict
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
#load model weights
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
#Set model to train mode
model.eval()
#Get previous random states
torch.set_rng_state(checkpoint["torch_rng_state"].to("cpu")),
torch.cuda.set_rng_state_all([checkpoint["cuda_rng_state"][0].to("cpu")]),
np.random.set_state(checkpoint["numpy_rng_state"]),
random.setstate(checkpoint["python_rng_state"])
g.set_state(checkpoint["generator_rng_state"].to("cpu"))

#Setting generator parameters
len_seq= 2**20#23#2**22 gives around 1.3 years, 2**23 around 2.6 years
dt=10.#10
fs=1/dt
T= len_seq*dt/round(YRSID_SI)#Not actually input into the generator, it already calculates this and stores as an attribute
TDI_channels="AE"
n_channels=len(TDI_channels)
add_noise=False#True#False
add_glitches=True
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
batch_size=1
test_size=0.3

#Initialise the dataset classes for training and val
EMRI_params_dir="training_data/EMRI_params.npy"
EMRI_params_and_SNRs= np.load(EMRI_params_dir, allow_pickle=True)
train_params, val_params= train_test_split(EMRI_params_and_SNRs, test_size=test_size, random_state=seed)

# training_set= EMRIGeneratorTDI(train_params, dim=len_seq, dt=dt, TDI_channels=TDI_channels,
#                                 add_noise=add_noise, add_glitches=add_glitches, seed=seed, target_SNR_range=training_target_SNR_range, use_gpu=use_cuda,
#                                 random_windows=random_windows,
#                                 wavelet_transform=wavelet_transform, Nt=Nt, Nf=Nf, nx=nx, mult=mult)
validation_set= EMRIGeneratorTDI(val_params, dim=len_seq, dt=dt, TDI_channels=TDI_channels,
                                  add_noise=add_noise, add_glitches=add_glitches, seed=seed, target_SNR_range=validation_target_SNR_range, use_gpu=use_cuda,
                                  random_windows=random_windows,
                                  wavelet_transform=wavelet_transform, Nt=Nt, Nf=Nf, nx=nx, mult=mult)

no_val_samples= 1024
validation_subset= torch.utils.data.Subset(validation_set, np.arange(no_val_samples).tolist())

#Get current states so we can reset the val set at each epoch
val_torch_rng_state = torch.get_rng_state()
val_cuda_rng_state = torch.cuda.get_rng_state_all()
val_numpy_rng_state = np.random.get_state()
val_python_rng_state = random.getstate()

#Set the current state for the EMRI generators
validation_set.set_seed_state(val_torch_rng_state, 
                            val_cuda_rng_state, 
                            val_numpy_rng_state)

#Initialise validation dataloader
'''Generator needs a fixed state with NO data augmentation'''
val_g = torch.Generator()
val_g.manual_seed(seed)
validation_dataloader= torch.utils.data.DataLoader(validation_subset, 
                                                   batch_size=batch_size, 
                                                   shuffle=True, 
                                                   drop_last=True, 
                                                   generator=val_g)

validation_dataloader= torch.utils.data.DataLoader(validation_subset, batch_size=batch_size, shuffle=True, drop_last=True, generator=g)

#Generate data
# with torch.inference_mode():
#     '''IMPORTANT: Using mitigated glitch backgrounds for the val. data!!!!'''
validation_set.load_glitch_background(glitch_background_dir="/fred/oz303/aboumerd/EMRI_Glitches/data_files/glitch_bg_AET/max_glitch_SNR_inf/",
                                                            background_idx=0)
X_EMRIs, y_true_EMRIs, residual, mask_EMRIs, mask_residual = next(iter(validation_dataloader))

'''
Going to do a saliency map
'''
#need to enable gradient logging on the input tensor
X_EMRIs.requires_grad_(True)
mask_EMRIs.requires_grad_(True)

#Forward pass
pred_masks, _ = model(X_EMRIs)

#Reverse preprocessing
#Define parameters for applying mask, and doing backtransforms                    
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
residual_bt = preprocess_signal(residual, 
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
#Calculate predicted backtransformed residual
pred_residual_bt = X_bt - pred_denoised_bt
#Calculate the predictions and residuals in the transformed domain
pred_denoised = preprocess_signal(pred_denoised_bt, 
                    asinh_scale=asinh_scale, 
                    # asinh_shift=asinh_shift, 
                    max_absolute=max_abs_tensor,
                    inverse = False
                    ).float()
pred_residual = preprocess_signal(pred_residual_bt, 
                    asinh_scale=asinh_scale, 
                    # asinh_shift=asinh_shift, 
                    max_absolute=max_abs_tensor,
                    inverse = False
                    ).float()


#Initialise loss function
'''Need to use the full loss! Not just the pixel-wise loss of the mask!'''
pixel_loss = SubBandCharbonnierLoss()
overlap_loss_fn = OverlapLoss(eps=1e-15)#eps=1e-4
gradient_loss_fn = GradientDifferenceLoss(eps=1e-16)
SNR_loss_fn = DetSNRLoss()
log_mag_loss = LogMagnitudeL1Loss(epsilon=1e-16)
opt_SNR_loss = OptSNRLoss(eps=1e-8)
XWT_power_loss = CrossWaveletPowerLoss(scale_factor=1.0, eps=1e-16)#eps=1e-8

#Compute loss
region_weighting = torch.abs(mask_EMRIs)
loss = pixel_loss(pred_masks, mask_EMRIs, region_weighting = region_weighting, normalise=False)
loss = loss + 0.1 * overlap_loss_fn(pred_denoised_bt, y_bt)
loss = loss + 0.1 * gradient_loss_fn(pred_denoised_bt, y_bt)
loss = loss + 0.01 * log_mag_loss(pred_denoised_bt, y_bt)#, region_weighting = region_weighting

print(f"Overlap of {1-overlap_loss_fn(pred_denoised_bt, y_bt)}")

#Need to retain loss grad
loss.retain_grad()

#Get gradients with a backwards pass
loss.backward()

#Calculate saliency map as max abs gradient across all channels
saliency, _ = torch.max(X_EMRIs.grad.data.abs(), dim=1)
# saliency = saliency.reshape(224, 224)

#Plot saliency map
time_grid, freq_grid = compute_bins(Nf, Nt, T*365*86400)
vmin= min(np.min(np.abs(X_bt[0,0].detach().numpy())), 
                 np.min(np.abs(pred_denoised_bt[0,0].detach().numpy())),
                 np.min(np.abs(pred_denoised_bt[0,0].detach().numpy())),
                 np.min(np.abs(y_bt[0,0].detach().numpy()))
                 )#0.

vmax= max(np.max(np.abs(X_bt[0,0].detach().numpy())), 
                 np.max(np.abs(pred_denoised_bt[0,0].detach().numpy())),
                 np.max(np.abs(pred_denoised_bt[0,0].detach().numpy())),
                 np.max(np.abs(y_bt[0,0].detach().numpy()))
                 )#0.

fig, axs= plt.subplots(3,1, figsize=(8.27, 11.69), sharex=True, layout='constrained')#, height_ratios=[0.5]
im0 = axs[0].imshow(np.abs(X_bt[0,0].detach().numpy()),
              cmap="inferno", 
              origin="lower", 
              norm="log",
              interpolation="nearest",
              extent=[time_grid[0], time_grid[-1], freq_grid[1], freq_grid[-1]],
              aspect='auto',
              vmin=vmin,
              vmax=vmax,
)#y_bt[0,0]#pred_denoised_bt[0,0]
im1 = axs[1].imshow(np.abs(pred_denoised_bt[0,0].detach().numpy()),
              cmap="inferno", 
              origin="lower", 
              norm="log",
              interpolation="nearest",
              extent=[time_grid[0], time_grid[-1], freq_grid[1], freq_grid[-1]],
              aspect='auto',
              vmin=vmin,
              vmax=vmax,
              )#y_bt[0,0]#pred_denoised_bt[0,0]
im2 = axs[2].imshow(np.abs(y_bt[0,0].detach().numpy()),
              cmap="inferno", 
              origin="lower", 
              norm="log",
              interpolation="nearest",
              extent=[time_grid[0], time_grid[-1], freq_grid[1], freq_grid[-1]],
              aspect='auto',
              vmin=vmin,
              vmax=vmax,
              )#y_bt[0,0]#pred_denoised_bt[0,0]
# im3 = axs[3].imshow(saliency[0].cpu(),
#               cmap="viridis", 
#               origin="lower", 
#               norm="log",
#               interpolation="nearest",
#               extent=[time_grid[0], time_grid[-1], freq_grid[1], freq_grid[-1]],
#               aspect='auto',
#             #   vmin=vmin,
#             #   vmax=vmax,
#              )#
axs[-1].set_xlabel("Time, seconds")
axs[-1].set_ylabel("Freq, Hz")
axs[0].set_ylabel("Freq, Hz")
axs[1].set_ylabel("Freq, Hz")
axs[2].set_ylabel("Freq, Hz")
plt.colorbar(im0, ax=axs[0], label="Absolute Wavelet Amplitude")
plt.colorbar(im1, ax=axs[1], label="Absolute Wavelet Amplitude")
plt.colorbar(im2, ax=axs[2], label="Absolute Wavelet Amplitude")

# plt.colorbar(im2, ax=axs[1], label="Absolute Wavelet Amplitude")
# axs[0].axis('off')
# axs[1].axis('off')
# axs[2].axis('off')
# axs[3].axis('off')
fig.savefig("test_saliency_map.pdf")



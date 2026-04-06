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

checkpoint_path = "model_checkpoint.tar"

#Load checkpoint dict
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
#load model weights
model.load_state_dict(checkpoint['model_state_dict'])
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

validation_dataloader= torch.utils.data.DataLoader(validation_subset, batch_size=batch_size, shuffle=True, drop_last=True, generator=g)

#Generate data
validation_set.load_glitch_background(glitch_background_dir="/fred/oz303/aboumerd/EMRI_Glitches/data_files/glitch_bg_AET/max_glitch_SNR_8.0/",
                                                            background_idx=0)
X_EMRIs, y_true_EMRIs, mask_EMRIs = next(iter(validation_dataloader))

'''
Going to do a saliency map
'''
#need to enable gradient logging on the input tensor
X_EMRIs.requires_grad_(True)
mask_EMRIs.requires_grad_(True)

#Forward pass
pred, denoised = model(X_EMRIs)

#Initialise loss function
'''Need to use the full loss! Not just the pixel-wise loss of the mask!'''
loss_fn = CharbonnierLoss()

#Compute loss
loss = loss_fn(pred, mask_EMRIs) + loss_fn(denoised, y_true_EMRIs)

#Need to retain loss grad
loss.retain_grad()

#Get gradients with a backwards pass
loss.backward()

#Calculate saliency map as max abs gradient across all channels
saliency, _ = torch.max(X_EMRIs.grad.data.abs(), dim=1)
# saliency = saliency.reshape(224, 224)

#Plot saliency map
fig, axs= plt.subplots(1,2, height_ratios=[0.5])
axs[0].imshow(np.abs(y_true_EMRIs[0,0].detach().numpy()),
              cmap="inferno", origin="lower")
axs[1].imshow(saliency[0].cpu(), cmap='hot',
             origin="lower")
axs[0].axis('off')
axs[1].axis('off')
fig.savefig("test_saliency_map.png")



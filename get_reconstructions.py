'''
This script is used to generate and save a dataset of testing data.
Testing data can include things such as:

1. LISA noise
2. Noisy EMRIs not seen by the model
3. Other types of GW sources e.g. MBHBs etc.
4. Glitches
'''

import numpy as np
import cupy as xp
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
from src.tools.test_and_train_loop import *
from src.tools.model_architecture import ConvAE
from few.utils.constants import YRSID_SI
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split


from src.tools.EMRI_generator_TDI import EMRIGeneratorTDI

# GPU check
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

#Specify some variables
model_state_dict_dir= "/fred/oz303/aboumerd/EMRI_denoising/model_best_performance.pt"

#Load model's weights and architecture
model= ConvAE().to(device)
model.load_state_dict(torch.load(model_state_dict_dir, weights_only=True, map_location=device))#, map_location=device
model.eval()

#Specify EMRI generator params
dim= 2**20#23#2**22 gives around 1.3 years, 2**23 around 2.6 years
dt=10#10
fs=1/dt
T= dim*dt/round(YRSID_SI)#Not actually input into the generator, it already calculates this and stores as an attribute
TDI_channels="AE"
n_channels=len(TDI_channels)
add_noise=True#False
seed=2023
batch_size=4
# training_target_SNR_range = [70, 80]
validation_target_SNR_range = [20, 80]

#Set some seeds
torch.manual_seed(seed)
g = torch.Generator()
g.manual_seed(seed)

#Initialise the dataset classes for training and val
EMRI_params_dir="/fred/oz303/aboumerd/EMRI_denoising/training_data/EMRI_params.npy"
EMRI_params= np.load(EMRI_params_dir, allow_pickle=True)
_, val_params= train_test_split(EMRI_params, test_size=0.3, random_state=seed)

validation_set= EMRIGeneratorTDI(val_params, dim=dim, dt=dt, TDI_channels=TDI_channels,
                                  add_noise=add_noise, seed=seed, target_SNR_range=validation_target_SNR_range, use_gpu=use_cuda)


#Initialise the data generators as PyTorch dataloaders
validation_dataloader= torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True,  generator=g)

#Generate one batch of data
X_EMRIs, y_true_EMRIs = next(iter(validation_dataloader))

#Normalise X
max_abs_tensor= 4.0
X_EMRIs= X_EMRIs/max_abs_tensor

#Make predictions with the model
y_pred_EMRIs= model(X_EMRIs)

#Convert everything to numpy arrays
X_EMRIs= X_EMRIs.detach().cpu().numpy()
y_true_EMRIs= y_true_EMRIs.detach().cpu().numpy()
y_pred_EMRIs= y_pred_EMRIs.detach().cpu().numpy()


#Save the example EMRIs and their reconstructions!
np.save("Val_X_EMRIs_NORMALISED.npy", X_EMRIs)
np.save("Val_y_true_EMRIs.npy", y_true_EMRIs)
np.save("Val_y_pred_EMRIs.npy", y_pred_EMRIs)

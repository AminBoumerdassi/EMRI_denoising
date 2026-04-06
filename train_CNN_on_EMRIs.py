'''
This script trains a CNN-autoencoder using the EMRI data generator, records losses, and plots them.
It also uses some custom callbacks for testing on noise at the end of each epoch.
'''

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
from src.tools.test_and_train_loop import *
from src.tools.model_architecture import *
from src.tools.losses import *
from src.tools.transforms import *

#For debugging (ONLY turn on when needed)
torch.autograd.set_detect_anomaly(False)

# Using mixed precision
'''Keeping mixed precision off for now as it seems to worsen performance.
   It needs a bit of finesse, and deeper reading into how best to apply
   it for this ML problem.'''
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
'''Just remember to convert input as well'''
model = model.to(memory_format=torch.channels_last)

#Trick: compile the model
model = torch.compile(model)

# Initialise loss function and optimizer
'''About learning rates:
For the timeseries EMRIs, keeping the lr to 14e-4.
This may be too large for the time-frequency data.
This could explain the bounce in loss. The lr is so high, that the weight updates
become large enough to miss or effectively orbit around the minimum.

Can try lr=14e-5, 14e-6, ... etc.

Decreasing the lr helped significantly but it may still be too high as the loss increases midway through the 1st epoch.

Now try 1e-4, and potentially lower. Want to completely eliminate the brief increase in loss.
'''

lr=5e-4#1e-4#1e-4#1e-5#1e-4#1e-5#14e-4##benchmark at 14e-4
'''Change of loss function: mae loss as it seems to do better at image denoising.'''
loss_fn= CharbonnierLoss().to(device)#nn.L1Loss().to(device)#nn.MSELoss().to(device)#CharbonnierLoss().to(device)#
'''Change of optimiser: from Adam to RAdam'''
optimizer= torch.optim.RAdam(params=model.parameters(), lr=lr)#torch.optim.Adam(params=model.parameters(), lr=lr)

#Defining Lr scheduler
'''Change of LR scheduler to reduce lr on plateau'''
scheduler = None#torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=25)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)#1000

#Doing pretraining? Set here
pretrained=True
pretraining_weights = "/fred/oz303/aboumerd/EMRI_denoising/pretraining/model_checkpoint.tar"
if pretrained:
    print("Using pretrained weights!")
    checkpoint = torch.load(pretraining_weights, map_location=device, weights_only=False)
    #Load pretrained weights
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print("Not using pretrained weights!")

# Continuing training? Set here!!!!!!!!
checkpoint_path = None#"model_checkpoint.tar"#None#"model_checkpoint.tar"#None#"model_checkpoint.tar"#None#"model_checkpoint.tar"#None#"model_checkpoint.tar"#

if checkpoint_path is not None:
    print("Continuing training!")
    #Load checkpoint dict
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    #load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    #Load optimiser state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #Load scheduler state
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    #Set the most recent losses
    train_history=[]
    val_history=[checkpoint['loss']]
    #Set model to train mode
    model.train()
    #Get previous random states
    torch.set_rng_state(checkpoint["torch_rng_state"].to("cpu")),
    torch.cuda.set_rng_state_all([checkpoint["cuda_rng_state"][0].to("cpu")]),
    np.random.set_state(checkpoint["numpy_rng_state"]),
    random.setstate(checkpoint["python_rng_state"])
    g.set_state(checkpoint["generator_rng_state"].to("cpu"))
else:
    print("Training from scratch!")
    start_epoch = 0
    #initialise training and validation histories
    train_history=[]
    val_history=[]

#Setting generator parameters
len_seq= 2**20#23#2**22 gives around 1.3 years, 2**23 around 2.6 years
dt=10.#10
fs=1/dt
T= len_seq*dt/round(YRSID_SI)#Not actually input into the generator, it already calculates this and stores as an attribute
TDI_channels="AE"
n_channels=len(TDI_channels)

'''No noise or glitches because of we are pretraining! Change later!'''
add_noise=False#True#False
add_glitches=True#True


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

#Initialise the dataset classes for training and val
EMRI_params_dir="training_data/EMRI_params.npy"
EMRI_params_and_SNRs= np.load(EMRI_params_dir, allow_pickle=True)
train_params, val_params= train_test_split(EMRI_params_and_SNRs, test_size=test_size, random_state=seed)

training_set= EMRIGeneratorTDI(train_params, dim=len_seq, dt=dt, TDI_channels=TDI_channels,
                                add_noise=add_noise, add_glitches=add_glitches, seed=seed, target_SNR_range=training_target_SNR_range, use_gpu=use_cuda,
                                random_windows=random_windows,
                                wavelet_transform=wavelet_transform, Nt=Nt, Nf=Nf, nx=nx, mult=mult)
validation_set= EMRIGeneratorTDI(val_params, dim=len_seq, dt=dt, TDI_channels=TDI_channels,
                                  add_noise=add_noise, add_glitches=add_glitches, seed=seed, target_SNR_range=validation_target_SNR_range, use_gpu=use_cuda,
                                  random_windows=random_windows,
                                  wavelet_transform=wavelet_transform, Nt=Nt, Nf=Nf, nx=nx, mult=mult)

#For the training data, we will only train on randomised subsets of the data per epoch for the sake of time
'''BTW this technique is called 'Repeated Sampling of Random Subsets' '''
samples_per_epoch = 1024
sampler = torch.utils.data.RandomSampler(training_set, num_samples=samples_per_epoch, replacement=False, generator=g)

#For the validation data, we will use a subset of the whole val. data
no_val_samples= 1024
validation_subset= torch.utils.data.Subset(validation_set, np.arange(no_val_samples).tolist())

#Initialise the data generators as PyTorch dataloaders
training_dataloader= torch.utils.data.DataLoader(training_set, batch_size=batch_size, sampler=sampler, shuffle=None, drop_last=True, generator=g)
validation_dataloader= torch.utils.data.DataLoader(validation_subset, batch_size=batch_size, shuffle=True, drop_last=True, generator=g)

#See the architecture of the model
summary(model, input_size=(batch_size, n_channels, Nf, Nt))#(batch_size, n_channels, Nf, Nt)

#Declare generator's parameters
training_set.declare_generator_params()

#Declare hyperparameters
print("#################################")
print("####TRAINING HYPERPARAMETERS####")
print("#Batch size: ", batch_size)
print("#Learning rate:", lr)
print("#Training proportion of dataset: ", 1-test_size)
print("#No. epochs: ", epochs)
print("#################################")

#Initialise a summary writer for tensorboard
comment=''
filename_suffix=''
writer = SummaryWriter(log_dir=None, filename_suffix=filename_suffix, comment=comment)

#Load example val. data without graphs or extra things we don't need
with torch.inference_mode():
    '''IMPORTANT: Using mitigated glitch backgrounds for the val. data!!!!'''
    validation_set.load_glitch_background(glitch_background_dir="/fred/oz303/aboumerd/EMRI_Glitches/data_files/glitch_bg_AET/max_glitch_SNR_8.0/",
                                                                background_idx=0)
    X_EMRIs, y_true_EMRIs, mask_EMRIs = next(iter(validation_dataloader))

#Restrict ourselves to the 1st sample in the batch
#And convert to channels last format
X_EMRIs = X_EMRIs[:1,:,:,:].to(memory_format=torch.channels_last)
y_true_EMRIs = y_true_EMRIs[:1,:,:,:].to(memory_format=torch.channels_last)
mask_EMRIs = mask_EMRIs[:1,:,:,:].to(memory_format=torch.channels_last)

# Preprocess/rescale example val. data input and targets
'''
Maximum of the glitches in A and E at SNRs <8.0 was array([37.57473783, 36.71401857])
Maximum of the loudest EMRIs in A and E was [30.990543, 30.842495]

It's possible that standardisation may be better suited to wavelet coefficients.
Let's load up a batch of training data, and plot a histogram. If we see somethhing
normally distributed then standardisation should be appropriate.

'''
max_abs_tensor= None#torch.as_tensor([30.990543, 30.842495], device=device)[None,:,None,None]#torch.as_tensor([37.57473783, 36.71401857], device=device)[None,:,None,None]
# X_transformed = asinh_transform(X_EMRIs, a=1 / 1e-13, c=0.0, forward=True)
# X_transformed= normalise(X_transformed, scale_factor=max_abs_tensor, forward=True)

# y_transformed = asinh_transform(y_true_EMRIs, a=1 / 1e-13, c=0.0, forward=True)
# y_transformed= normalise(y_transformed, scale_factor=max_abs_tensor, forward=True)

# corruption = X_EMRIs - y_true_EMRIs
# corruption_transformed = asinh_transform(corruption, a=1 / 1e-13, c=0.0, forward=True)
# corruption_transformed= normalise(corruption_transformed, scale_factor=max_abs_tensor, forward=True)

##Calculate a mask
#mask_EMRI = mask_EMRIs#y_transformed/(X_transformed + 1e-4)
#mask_glitches = corruption_transformed/X_transformed
#masked_data= mask_EMRI * X_EMRIs#X_transformed

##Let's do a histogram of the mask values
#plt.figure()
#plt.imshow(np.abs(mask_EMRI[0,0,:,:].detach().numpy()))
#plt.imshow(torch.abs((mask_EMRI*X_EMRIs)[0,0,:,:]).detach().numpy(), cmap="inferno", origin="lower", vmin=0., vmax=1.0)
#plt.colorbar()
#plt.hist(mask_EMRI.detach().numpy()[0,0].flatten(), bins=100, log=True)
#plt.savefig("mask_hist.png")


# '''To correctly calculate the "best fit":'''
#bestfit= X_transformed - y_transformed#X_transformed - corruption_transformed

# X_EMRIs = torch.asinh(X_EMRIs/1e-13)/max_abs_tensor
# y_true_EMRIs = torch.asinh(y_true_EMRIs/1e-13)/max_abs_tensor


# Convert target to np array for later calculations and plotting
# y_true_EMRIs= y_true_EMRIs.detach().cpu().numpy()

#validation_set.retune_glitches(1.0)#1e-9
# validation_set.load_glitch_background(glitch_background_dir="/fred/oz303/aboumerd/EMRI_Glitches/data_files/glitch_bg_AET/max_glitch_SNR_8.0/",
#                                                             background_idx=0)


'''
#Plot of example val data
plt.figure()
time_grid, freq_grid = compute_bins(Nf, Nt, T*365*86400)
plt.title("Input validation data: Glitchy EMRI")
plt.xlabel("Time, days")
plt.ylabel("Freq, Hz")
plt.imshow(np.abs(X_EMRIs[0,0,:,:].cpu().numpy()),
                    cmap="inferno", origin="lower",
                    interpolation="nearest",
                    extent=[time_grid[0], time_grid[-1], freq_grid[0], freq_grid[-1]],
                    aspect="auto",
                    # vmax=1.0,
                    vmin=0.0
                    #norm="log",
                    )
plt.colorbar(label="Rescaled strain")
plt.savefig("val_EMRI_tf.png")
'''

'''
Example loss calculation
for i in range(0,5):
    max_abs_tensor= 30.0#for arcsinh transform
    X_EMRIs, y_true_EMRIs = next(iter(validation_dataloader))
    X_EMRIs = torch.asinh(X_EMRIs/1e-13)/max_abs_tensor
    y_pred_EMRIs= model(X_EMRIs)#.to(torch.float32)
    loss_fn(y_true_EMRIs, X_EMRIs)#.to(torch.float32), .to(torch.float32)
'''

#Define a epoch-wise schedule for the glitch rescaling
glitch_scale_schedule = np.geomspace(1e-9, 1.0, num=100)
#Define a randomised order of glitch backgrounds
glitch_background_idxs = np.random.choice(np.arange(1,128), size=epochs-start_epoch)

#Define a scheduler for the minimum SNR range
SNR_min=20.0
SNR_max=80.0
schedule_length=200
def min_SNR_scheduler(epoch, SNR_min=20.0, SNR_max=80.0, schedule_length=120):
    '''A cosine-based scheduler to smoothly increase the SNR range.'''
    if epoch > schedule_length:
        out = SNR_min
    else:
        out = float(SNR_max - 0.5*(SNR_max-SNR_min)*(1-np.cos(np.pi*epoch/schedule_length)))
    # #Make sure anything past the end of the schedule is set to the min SNR
    # out[schedule_length:] = SNR_min
    return out


#Train the model
for t in range(start_epoch, epochs):
    #Initialise variables for measuring time of an epoch
    start= torch.cuda.Event(enable_timing=True)
    end= torch.cuda.Event(enable_timing=True)
    print(f"-----------------------------------\n\t\tEpoch {t+1}/{epochs}\n-----------------------------------")
    #Scheduling the SNR range!
    # current_min_SNR = min_SNR_scheduler(t, SNR_min=SNR_min, SNR_max=SNR_max, schedule_length=schedule_length)
    # print(f"Tuning min. SNR to: {current_min_SNR}")
    # training_set.retune_SNRs(low=current_min_SNR, high=SNR_max)
    training_set.retune_SNRs(low=validation_target_SNR_range[0], high=validation_target_SNR_range[1])
    #Load a randomised glitch background at each epoch
    print(f"Using glitch BG no. {glitch_background_idxs[t]} at SNR threshold 8")
    training_set.load_glitch_background(glitch_background_dir="/fred/oz303/aboumerd/EMRI_Glitches/data_files/glitch_bg_AET/max_glitch_SNR_8.0/",
                                                            background_idx=glitch_background_idxs[t])
    start.record()
    train_loop(training_dataloader, model, loss_fn, optimizer, batch_size, train_history, scaler, str(device), use_amp=use_amp, max_abs_tensor=max_abs_tensor)
    # Log gradients (e.g., at the end of each epoch)
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f'{name}.grad', param.grad, t)
    val_loop(validation_dataloader, model, loss_fn, val_history, scaler, str(device), use_amp=use_amp, max_abs_tensor=max_abs_tensor)
    #Step the optimizer after each val loop
    if scheduler is not None:
        scheduler.step(val_history[-1])
    else:
        pass
    end.record()

    #Print time for 1 epoch
    torch.cuda.synchronize()
    print("Epoch time: {:.2f}s\n".format(start.elapsed_time(end)/1000))

    #Save model state dict if lowest loss achieved
    if val_history[-1] == np.array(val_history).min():
        best_model_state_dict = copy.deepcopy(model.state_dict())
        torch.save(best_model_state_dict, "model_best_performance.pt")
    #Save a checkpoint for further training if needed
    checkpoint = {
        'epoch': t,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'scheduler_state_dict': scheduler.state_dict(),
        'loss': val_history[-1],
        #For reproducibility
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "generator_rng_state": g.get_state()
    }
    torch.save(checkpoint, "model_checkpoint.tar")
    
    #Save the current history
    '''np.save("train_history_current.npy",train_history)
    np.save("val_history_current.npy", val_history)'''
    #Get an example val. reconstruction at this particular epoch
    '''The model output is now the glitches without the EMRI!'''
    #Do val. forward pass in inference mode and mixed precision
    with torch.inference_mode():
        with torch.autocast(device_type=str(device), dtype=torch.bfloat16, enabled=use_amp):
            _, y_pred_EMRIs= model(X_EMRIs)
    #convert prediction to np array
    y_pred_EMRIs= y_pred_EMRIs.float().cpu().numpy()
    #Plot the reconstruction and residual
    if wavelet_transform:
        reconstruction_fig, axs = plt.subplots(1,2, sharey=True, figsize=(10, 5),layout='constrained')
        time_grid, freq_grid = compute_bins(Nf, Nt, T*365*86400)
        vmin=0.
        vmax=1.0
        reconstruction_fig.suptitle("Val. data reconstruction in A chan. at epoch {:}".format(t+1))
        axs[0].set_title("Masked data")
        axs[1].set_title("Clean data")
        axs[0].set_xlabel("Time, seconds")
        axs[1].set_xlabel("Time, seconds")
        axs[0].set_ylabel("Freq, Hz")
        im1 = axs[0].imshow(np.abs(y_pred_EMRIs)[0,0,:,:],#[0,0,:,:]#X_transformed.detach().cpu().numpy() - np.arctanh(y_pred_EMRIs)
                            cmap="inferno", origin="lower",
                            interpolation="nearest",
                            extent=[time_grid[0], time_grid[-1], freq_grid[1], freq_grid[-1]],
                            vmin=vmin,
                            vmax=vmax,
                            aspect='auto',
                            #norm="log",
                            )
        im2 = axs[1].imshow(np.abs(y_true_EMRIs.detach().cpu().numpy())[0,0,:,:],#[0,0,:,:]#y_pred_EMRIs#best_fit_transformed
                            cmap="inferno", origin="lower",
                            interpolation="nearest",
                            extent=[time_grid[0], time_grid[-1], freq_grid[1], freq_grid[-1]],
                            vmin=vmin,
                            vmax=vmax,
                            aspect='auto',
                            # norm="log",
                            )
        # plt.colorbar(im1, ax=axs[0], label="Rescaled strain")
        # plt.colorbar(im2, ax=axs[1], label="Rescaled strain")
        reconstruction_fig.colorbar(im2, label="Rescaled strain", ax=axs.ravel().tolist())#, shrink=0.75
        # reconstruction_fig.tight_layout()
        # axs[0].set_box_aspect(1)
        # axs[1].set_box_aspect(1)
        #plt.savefig("test_val.png")
    else:
        reconstruction_fig = plt.figure()
        #Time-domain reconstruction
        # plt.title("Val. data reconstruction in A chan. at epoch {:}".format(t+1))
        # plt.xlabel("Time, years")
        # plt.ylabel("Strain")
        # plt.plot(np.linspace(0, T, num=len_seq)[:150000], y_true_EMRIs[0,0,:150000], label="Target")
        # plt.plot(np.linspace(0, T, num=len_seq)[:150000], y_pred_EMRIs[0,0,:150000], label="Prediction")
        # plt.plot(np.linspace(0, T, num=len_seq)[:150000], residual[0,0,:150000], label="residual")
        # plt.legend(loc='upper right')
    #Save and close fig
    plt.savefig("reconstructions_td_live_epoch_{:}.png".format(t+1), bbox_inches="tight")#
    plt.close()

    #Record training, val. losses, and reconstructions for tensorboard
    writer.add_scalar("Loss/train", train_history[t-start_epoch], t)
    writer.add_scalar("Loss/validation", val_history[t-start_epoch], t)
    writer.add_figure("Reconstruction_fig", reconstruction_fig, global_step=t)

#Close the tensorboard writer
writer.flush()
writer.close()
print("Done!")


#Save the training and val losses
'''EDIT THESE FILE NAMES!'''
# np.save("train_history.npy",train_history)
# np.save("val_history.npy", val_history)

#Save model
'''EDIT AS NEEDED!'''
#torch.save(model.state_dict(), "model_BS_{:}_lr_0_{:}.pt".format(batch_size, str(lr)[2:]))
#torch.save(model.state_dict(), "model_current.pt")
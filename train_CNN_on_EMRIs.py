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
model= Dilated_ConvAE().to(device)#ConvAE().to(device)

# Make model in channels-last format for speedup
model = model.to(memory_format=torch.channels_last)

#Trick: compile the model
model = torch.compile(model)

#Wrap model into EMA object. Use the Switch-EMA variant.
ema = EMA(
    model,
    beta = 0.999,#0.9999,              # exponential moving average factor
    update_after_step = 5000,#100,    # only after this number of .update() calls will it start updating
    update_every = 1,#10,          # how often to actually update, to save on compute (updates every 10th .update() call)
)

#Ensure that only weights are being decayed
decay = []
no_decay = []

for name, param in model.named_parameters():

    if not param.requires_grad:
        continue

    # Exclude biases and 1D params
    if param.ndim == 1 or name.endswith(".bias") or "alpha" in name:
        no_decay.append(param)
    else:
        decay.append(param)

#Initialise autoclip object
clipper = QuantileClip(model.parameters(), quantile=0.9, history_length=100)#quantile=0.9,history_length=1000

# Initialise loss function and optimizer
lr=3e-3#5e-3#5e-4#1e-4#1e-4#1e-5#1e-4#1e-5#14e-4##benchmark at 14e-4
weight_decay = 0.0#1e-5
loss_fn= SubBandCharbonnierLoss().to(device)#nn.L1Loss().to(device)#nn.MSELoss().to(device)#CharbonnierLoss().to(device)#
optimizer= torch.optim.RAdam(params=[
                            {"params": decay, "weight_decay": weight_decay},
                            {"params": no_decay, "weight_decay": 0.0},
                            ], 
                             lr=lr,
                             decoupled_weight_decay=True)#torch.optim.Adam(params=model.parameters(), lr=lr)

#Defining Lr scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-5)#1000#torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=25)#

#Doing pretraining? Set here
pretrained=False#True
pretraining_weights = "/fred/oz303/aboumerd/EMRI_denoising/trained_nearly_model_with_CBAM_and_ASPP/model_checkpoint.tar"
if pretrained:
    print("Using pretrained weights!")
    checkpoint = torch.load(pretraining_weights, map_location=device, weights_only=False)
    #Load pretrained weights
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print("Not using pretrained weights!")

# Continuing training? Set here!!!!!!!!
checkpoint_path = None#"model_checkpoint.tar"#"/fred/oz303/aboumerd/EMRI_denoising/trained_nearly_model_with_CBAM_and_ASPP/model_checkpoint.tar"#"model_checkpoint.tar"#None#"model_checkpoint.tar"#None#"model_checkpoint.tar"#None#"model_checkpoint.tar"#None#"model_checkpoint.tar"#

if checkpoint_path is not None:
    print("Continuing training!")
    #Load checkpoint dict
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    #load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    #Load optimiser state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #Load scheduler state
    '''NOTE: In previous jobs, the scheduler was mis-stepping according to the val loss rather than the epoch.
    So scheduler.last_epoch may need to be manually edited beforehand to fix this!'''
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #Load scaler state
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    #Set the most recent losses
    train_history=[]
    val_history=[checkpoint['loss']]
    #Set the most recent metrics
    train_pred_overlap_history=[]
    val_pred_overlap_history=[checkpoint['pred_overlap_history']]
    val_pred_overlap_history_EMA=[checkpoint['pred_overlap_history_EMA']]
    input_overlap_history=[checkpoint['input_overlap_history']]
    train_residual_overlap_history=[]
    #Set model to train mode
    model.train()
    #Get previous random states
    torch.set_rng_state(checkpoint["torch_rng_state"].to("cpu")),
    torch.cuda.set_rng_state_all([checkpoint["cuda_rng_state"][0].to("cpu")])
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    g.set_state(checkpoint["generator_rng_state"].to("cpu"))
else:
    print("Training from scratch!")
    start_epoch = 0
    #initialise training and validation histories
    train_history=[]
    val_history=[]
    val_history_EMA=[]
    #Initialise metric histories
    train_pred_overlap_history=[]
    val_pred_overlap_history=[]
    val_pred_overlap_history_EMA=[]
    input_overlap_history=[]
    train_residual_overlap_history=[]
    val_residual_overlap_history=[]
    val_residual_overlap_history_EMA=[]


#Setting generator parameters
len_seq= 2**20#23#2**22 gives around 1.3 years, 2**23 around 2.6 years
dt=10.#10
fs=1/dt
T= len_seq*dt/round(YRSID_SI)#Not actually input into the generator, it already calculates this and stores as an attribute
TDI_channels="AE"
n_channels=len(TDI_channels)

'''No noise or glitches because of we are pretraining! Change later!'''
add_noise=False
add_glitches=True#True
random_sign_flips=True


training_target_SNR_range = [20, 80]
validation_target_SNR_range = [20, 80]
random_windows=False#True

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

#See the architecture of the model
summary(model, input_size=(batch_size, n_channels, len_seq))#(batch_size, n_channels, Nf, Nt)

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
log_dir = "/fred/oz303/aboumerd/EMRI_denoising/runs_final/"
logfolder = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.mkdir(log_dir+logfolder)
comment='conv-tasnet'
filename_suffix='res_approach'
writer = SummaryWriter(log_dir= log_dir+logfolder, 
                       filename_suffix=filename_suffix, 
                       comment=comment)

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


#Train the model
for t in range(start_epoch, epochs):
    print(f"-----------------------------------\n\t\tEpoch {t+1}/{epochs}\n-----------------------------------")
    #Initialise variables for measuring time of an epoch
    start= torch.cuda.Event(enable_timing=True)
    end= torch.cuda.Event(enable_timing=True)
    #Apply per-epoch data augmentations on training but NOT validation set
    training_set.retune_SNRs(low=validation_target_SNR_range[0], high=validation_target_SNR_range[1])
    training_glitch_background_idx = np.random.randint(1,128)
    print(f"Training set has glitch BG no. {training_glitch_background_idx} at SNR threshold inf")
    training_set.load_glitch_background(glitch_background_dir=glitch_background_dir,
                                                        background_idx=training_glitch_background_idx)
    #Start training loop    
    start.record()
    train_loop(training_dataloader, 
               model, 
               loss_fn, 
               optimizer, 
               batch_size, 
               train_history, 
               scaler, 
               str(device), 
               use_amp=use_amp, 
               max_abs_tensor=None, 
               ema=ema,
               pred_overlap_history=train_pred_overlap_history,
               residual_overlap_history=train_residual_overlap_history,
               clipper=clipper,
               )#model
    #Reset the seed and state of the validation dataloader before each loop so the validation data doesn't change
    val_g.manual_seed(seed)
    validation_set.set_seed_state(val_torch_rng_state, 
                                val_cuda_rng_state, 
                                val_numpy_rng_state)
    #Start val. loop
    '''Not using ema model for the val loop'''
    val_loop(validation_dataloader, 
             model, 
             loss_fn, 
             val_history, 
             scaler, 
             str(device), 
             use_amp=use_amp, 
             max_abs_tensor=None, 
             pred_overlap_history=val_pred_overlap_history, 
             input_overlap_history=input_overlap_history,
             ema=ema,
             val_history_EMA=val_history_EMA,
             pred_overlap_history_EMA=val_pred_overlap_history_EMA,
             residual_overlap_history=val_residual_overlap_history,
             residual_overlap_history_EMA=val_residual_overlap_history_EMA,
             )
    #Step the scheduler after each val loop
    if scheduler is not None: scheduler.step()#Can need extra arguments depending on the scheduler
    # Log gradients at the end of each epoch
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f'{name}.grad', torch.nan_to_num(param.grad), t)
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
        'ema_model_state_dict': ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': val_history[-1],
        'loss_EMA': val_history_EMA[-1],
        "pred_overlap_history" : val_pred_overlap_history[-1],
        "pred_overlap_history_EMA" : val_pred_overlap_history_EMA[-1],
        "input_overlap_history" : input_overlap_history[-1],
        #For reproducibility
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "generator_rng_state": g.get_state()
    }
    torch.save(checkpoint, "model_checkpoint.tar")
    # #Save EMA model
    # torch.save(ema, "model_ema_checkpoint.tar")
    
    #Save the current history
    '''np.save("train_history_current.npy",train_history)
    np.save("val_history_current.npy", val_history)'''
    #Get an example val. reconstruction at this particular epoch
    '''The model output is now the glitches without the EMRI!'''
    #Do val. forward pass in inference mode and mixed precision
    with torch.inference_mode():
        with torch.autocast(device_type=str(device), dtype=torch.bfloat16, enabled=use_amp):
            '''Not using ema model for testing'''
            pred_signals = model(X_EMRIs)

    #convert prediction to np array
    y_pred_EMRIs= y_pred_EMRIs.float().cpu().numpy()
    #compute wavelet transform or stft of prediction 
    y_pred_EMRIs_wavelet = None#np.abs(y_pred_EMRIs)
    #Plot the reconstruction in the wavelet or stft domain
    reconstruction_fig, axs = plt.subplots(1,2, sharey=True, figsize=(10, 5),layout='constrained')
    reconstruction_fig.suptitle("Val. data reconstruction in A chan. at epoch {:}".format(t+1))
    axs[0].set_title("Masked data")
    axs[1].set_title("Clean data")
    axs[0].set_xlabel("Time, seconds")
    axs[1].set_xlabel("Time, seconds")
    axs[0].set_ylabel("Freq, Hz")
    # time_grid, freq_grid = compute_bins(Nf, Nt, T*365*86400)
    # vmin= min(np.min(y_pred_EMRI_scalogram), np.min(y_true_EMRI_scalogram))#0.
    # vmax= max(np.max(y_pred_EMRI_scalogram), np.max(y_true_EMRI_scalogram))#1.0
    # im1 = axs[0].imshow(y_pred_EMRI_scalogram[0,0,:,:],#[0,0,:,:]#X_transformed.detach().cpu().numpy() - np.arctanh(y_pred_EMRIs)
    #                     cmap="inferno", 
    #                     origin="lower",
    #                     interpolation="nearest",
    #                     extent=[time_grid[0], time_grid[-1], freq_grid[1], freq_grid[-1]],
    #                     vmin=vmin,
    #                     vmax=vmax,
    #                     aspect='auto',
    #                     norm="log",
    #                     )
    # im2 = axs[1].imshow(y_true_EMRI_scalogram[0,0,:,:],#[0,0,:,:]#y_pred_EMRIs#best_fit_transformed.detach().cpu().numpy()
    #                     cmap="inferno", 
    #                     origin="lower",
    #                     interpolation="nearest",
    #                     extent=[time_grid[0], time_grid[-1], freq_grid[1], freq_grid[-1]],
    #                     vmin=vmin,
    #                     vmax=vmax,
    #                     aspect='auto',
    #                     norm="log",
    #                     )
    #plt.colorbar(im2, ax=axs[1], label="Absolute Wavelet Amplitude")
    #Save and close fig
    plt.savefig("reconstructions_td_live_epoch_{:}.png".format(t+1), bbox_inches="tight")#
    plt.close()

    #Record training, val. losses, and reconstructions for tensorboard
    writer.add_scalar("Loss/train", train_history[t-start_epoch], t)
    writer.add_scalar("Loss/validation", val_history[t-start_epoch], t)
    writer.add_scalar("Loss/validation_EMA", val_history_EMA[t-start_epoch], t)
    writer.add_scalar("Metric/train_pred_overlap", train_pred_overlap_history[t-start_epoch], t)
    writer.add_scalar("Metric/train_residual_overlap", train_residual_overlap_history[t-start_epoch], t)
    writer.add_scalar("Metric/val_pred_overlap", val_pred_overlap_history[t-start_epoch], t)
    writer.add_scalar("Metric/val_pred_overlap_EMA", val_pred_overlap_history_EMA[t-start_epoch], t)
    writer.add_scalar("Metric/input_overlap", input_overlap_history[t-start_epoch], t)
    writer.add_scalar("Metric/val_residual_overlap", val_residual_overlap_history[t-start_epoch], t)
    writer.add_scalar("Metric/val_residual_overlap_EMA", val_residual_overlap_history_EMA[t-start_epoch], t)
    writer.add_figure("Reconstruction_fig", reconstruction_fig, global_step=t)

#Close the tensorboard writer
writer.flush()
writer.close()
print("Done!")
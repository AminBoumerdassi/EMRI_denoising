'''
This script trains a CNN-autoencoder using the EMRI data generator, records losses, and plots them.
It also uses some custom callbacks for testing on noise at the end of each epoch.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from few.utils.constants import YRSID_SI
import matplotlib.pyplot as plt
import os
import copy
from sklearn.model_selection import train_test_split
#from custom_callbacks import TestOnNoise

from src.tools.EMRI_generator_TDI import EMRIGeneratorTDI
from src.tools.test_and_train_loop import *
from src.tools.model_architecture import *

# GPU check
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#Using mixed precision
'''Keeping mixed precision off for now as it seems to worsen performance.
   It needs a bit of finesse, and deeper reading into how best to apply
   it for this ML problem.'''
use_amp=False#True
scaler = torch.amp.GradScaler(device ,enabled=use_amp)

#Initialise the model and move it to device
model= Dilated_ConvAE().to(device)#ConvAE().to(device)

#Setting generator parameters
len_seq= 2**20#23#2**22 gives around 1.3 years, 2**23 around 2.6 years
dt=10#10
fs=1/dt
T= len_seq*dt/round(YRSID_SI)#Not actually input into the generator, it already calculates this and stores as an attribute
TDI_channels="AE"
n_channels=len(TDI_channels)
add_noise=False#True#False
seed=2023
training_target_SNR_range = [70, 80]
validation_target_SNR_range = [20, 80]

#Setting training hyperparameters
batch_size=32
epochs=100
lr=14e-4#14e-4##benchmark at 14e-4
test_size=0.3

#Force PyTorch and numpy to be deterministic
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # numpy random generator
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
g = torch.Generator()
g.manual_seed(seed)

#Define loss functions and optimizer
loss_fn= nn.MSELoss().to(device)
optimizer= torch.optim.Adam(params=model.parameters(), lr=lr)

#Initialise the dataset classes for training and val
EMRI_params_dir="training_data/EMRI_params.npy"
EMRI_params_and_SNRs= np.load(EMRI_params_dir, allow_pickle=True)
train_params, val_params= train_test_split(EMRI_params_and_SNRs, test_size=test_size, random_state=seed)

training_set= EMRIGeneratorTDI(train_params, dim=len_seq, dt=dt, TDI_channels=TDI_channels,
                                add_noise=add_noise, seed=seed, target_SNR_range=training_target_SNR_range, use_gpu=use_cuda)
validation_set= EMRIGeneratorTDI(val_params, dim=len_seq, dt=dt, TDI_channels=TDI_channels,
                                  add_noise=add_noise, seed=seed, target_SNR_range=validation_target_SNR_range, use_gpu=use_cuda)

#Initialise the data generators as PyTorch dataloaders
training_dataloader= torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, drop_last=True, generator=g)
validation_dataloader= torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, drop_last=True, generator=g)

#See the architecture of the model
summary(model, input_size=(batch_size, n_channels, len_seq))

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

#Initialise callbacks
#TestOnNoise= TestOnNoise(model, training_and_validation_generator)

#initialise training and validation histories
train_history=[]
val_history=[]

#Initialise a summary writer for tensorboard
comment=''
filename_suffix=''
writer = SummaryWriter(log_dir=None, filename_suffix=filename_suffix, comment=comment)

#Load example val. data, and preprocess it
max_abs_tensor= 4.0
X_EMRIs, y_true_EMRIs = next(iter(validation_dataloader))
X_EMRIs = X_EMRIs/max_abs_tensor

#Train the model
for t in range(epochs):
    #Initialise variables for measuring time of an epoch
    start= torch.cuda.Event(enable_timing=True)
    end= torch.cuda.Event(enable_timing=True)
    print(f"-----------------------------------\n\t\tEpoch {t+1}/{epochs}\n-----------------------------------")
    #Here we can apply curriculum learning e.g. by changing the SNRs after a particular epoch, or turning on noise
    # #Anti-curriculum
    # if t in range(0,40):
    #     print("Retuning SNRs!")
    #     training_set.retune_SNRs(low=20, high=40)
    # if t in range(40,60):
    #     print("Retuning SNRs!")
    #     training_set.retune_SNRs(low=20, high=60)
    # if t in range(60,80):
    #     print("Retuning SNRs!")
    #     training_set.retune_SNRs(low=20, high=80)
    #Curriculum
    if t in range(0,40):
        print("Retuning SNRs!")
        training_set.retune_SNRs(low=60, high=80)
    if t in range(40,60):
        print("Retuning SNRs!")
        training_set.retune_SNRs(low=40, high=80)
    if t in range(60,80):
        print("Retuning SNRs!")
        training_set.retune_SNRs(low=20, high=80)
    start.record()
    train_loop(training_dataloader, model, loss_fn, optimizer, batch_size, train_history, scaler, "cuda", use_amp=use_amp)
    val_loop(validation_dataloader, model, loss_fn, val_history, scaler,  "cuda", use_amp=use_amp)
    end.record()

    #Print time for 1 epoch
    torch.cuda.synchronize()
    print("Epoch time: {:.2f}s\n".format(start.elapsed_time(end)/1000))

    #Save model if lowest loss achieved
    if val_history[-1] == np.array(val_history).min():
        best_model_state_dict = copy.deepcopy(model.state_dict())
        torch.save(best_model_state_dict, "model_best_performance.pt")
    
    #Save the current history
    '''np.save("train_history_current.npy",train_history)
    np.save("val_history_current.npy", val_history)'''

    #Quick plot of reconstructions at this particular epoch
    y_pred_EMRIs= model(X_EMRIs)
    reconstruction_fig = plt.figure()
    plt.title("Val. data reconstruction at epoch {:}".format(t+1))
    plt.xlabel("Time, years")
    plt.ylabel("Strain")
    plt.plot(np.linspace(0, T, num=len_seq)[:150000], y_true_EMRIs[0,0,:150000].detach().cpu().numpy(), label="Target")
    plt.plot(np.linspace(0, T, num=len_seq)[:150000], y_pred_EMRIs[0,0,:150000].detach().cpu().numpy(), label="Prediction")
    plt.plot(np.linspace(0, T, num=len_seq)[:150000], y_true_EMRIs[0,0,:150000].detach().cpu().numpy()-y_pred_EMRIs[0,0,:150000].detach().cpu().numpy(), label="residual")
    plt.legend(loc='upper right')
    plt.savefig("reconstructions_td_live_epoch_{:}.png".format(t+1))
    plt.close()

    #Record training, val. losses, and reconstructions for tensorboard
    writer.add_scalar("Loss/train", train_history[t], t)
    writer.add_scalar("Loss/validation", val_history[t], t)
    # writer.add_audio("Reconstruction/target", y_true_EMRIs[0,0,:150000], global_step=t, sample_rate=fs)
    # writer.add_audio("Reconstruction/prediction", y_pred_EMRIs[0,0,:150000], global_step=t, sample_rate=fs)
    # writer.add_audio("Reconstruction/residual", y_pred_EMRIs[0,0,:150000]-y_pred_EMRIs[0,0,:150000], global_step=t, sample_rate=fs)
    writer.add_figure("Reconstruction_fig", reconstruction_fig, global_step=t)

#Close the tensorboard writer
writer.flush()
writer.close()
print("Done!")


#Save the training and val losses
'''EDIT THESE FILE NAMES!'''
np.save("train_history.npy",train_history)
np.save("val_history.npy", val_history)
# np.save("train_history_BS_{:}_lr_0_{:}.npy".format(batch_size, str(lr)[2:]),train_history)
# np.save("val_history_BS_{:}_lr_0_{:}.npy".format(batch_size, str(lr)[2:]), val_history)

#Save model
'''EDIT AS NEEDED!'''
#torch.save(model.state_dict(), "model_BS_{:}_lr_0_{:}.pt".format(batch_size, str(lr)[2:]))
#torch.save(model.state_dict(), "model_current.pt")
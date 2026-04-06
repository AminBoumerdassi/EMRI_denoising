'''
This script is used to plot already generated reconstructions from the
get_reconstructions script.
'''

import numpy as np
import matplotlib.pyplot as plt

from few.utils.constants import YRSID_SI
import matplotlib.pyplot as plt
import os

X_EMRIs_fname= "/fred/oz303/aboumerd/EMRI_denoising/experiments/TCN_with_transpose_convs/Val_X_EMRIs_NORMALISED.npy"
y_pred_EMRIs_fname= "/fred/oz303/aboumerd/EMRI_denoising/experiments/TCN_with_transpose_convs/Val_y_pred_EMRIs.npy"
y_true_EMRIs_fname= "/fred/oz303/aboumerd/EMRI_denoising/experiments/TCN_with_transpose_convs/Val_y_true_EMRIs.npy"

#Load data
X_EMRIs= np.load(X_EMRIs_fname, allow_pickle=True) 
y_pred_EMRIs= np.load(y_pred_EMRIs_fname, allow_pickle=True)
y_true_EMRIs= np.load(y_true_EMRIs_fname, allow_pickle=True)

#Calculate the residuals
residuals =  y_true_EMRIs - y_pred_EMRIs

# #Load the normalising arrays
# max_abs_tensor= 4.0
# X_EMRIs= X_EMRIs*max_abs_tensor

#Define dataset parameters
dt=10
no_pts= X_EMRIs.shape[2]
T= X_EMRIs.shape[2]*10/YRSID_SI#years
t= np.linspace(0, T, num=no_pts)

#Plot the Y true, Y predicted, and residuals
'''Do something like 2 rows, 3 columns. Row 1 is for the A channel, row 2 the E channel'''
ncols=X_EMRIs.shape[0]
nrows= X_EMRIs.shape[1]

fig, axs= plt.subplots(nrows=nrows, ncols=ncols, sharex=True,gridspec_kw={"hspace":0})#(ax1, ax2, ax3, ax4, ax5, ax6)
fig.set_size_inches(15, 8)

#Plot inputs, predictions and residuals for each column of the subplot
for col in range(ncols):#subplot, axs.flatten()
  axs[0,col].set(title="EMRI "+ str(col+1))
  axs[-1,col].set_xlabel("Time, years")
  #If doing reconstructions across the A and E channels:
  for channel in range(nrows):
    # axs[channel,col].plot(t, X_EMRIs[col,channel,:], "b", label="Input", alpha=1)
    # axs[channel,col].plot(t, y_true_EMRIs[col,channel,:], "b", label="True EMRI", alpha=1)
    # axs[channel,col].plot(t, y_pred_EMRIs[col,channel,:], "r", label="Pred. EMRI", alpha=0.6)
    axs[channel,col].plot(t, residuals[col,channel,:], "g", label="Residual")

# #And label the subplots
fig.suptitle('Reconstructions of validation EMRI in TDI AE, SNRs [60,100]')
axs[0,0].set(ylabel="Whitened strain, A chan.")
axs[1,0].set(ylabel="Whitened strain, E chan.")
plt.legend()

plt.savefig("testing_data_td_reconstructions.png")

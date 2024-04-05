'''
This script contains custom callbacks such as one that makes the model test on LISA noise at the end of each epoch.
'''
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.experimental.dlpack import from_dlpack, to_dlpack
import cupy as xp
import numpy as np
from numpy.random import default_rng

class TestOnNoise(Callback):
    def __init__(self, model, generator):
        self.model = model
        self.generator = generator
        self.rng = default_rng(seed=2022)#Where do we use this in the current code??
        self.losses = []
        
    def on_epoch_end(self, epoch, logs={}):
        #Initialise an empty array to store noise samples for ONE batch
        x_test = xp.empty((self.generator.batch_size, self.generator.n_channels, self.generator.dim))
        #y_true= np.zeros((self.generator.batch_size, self.generator.n_channels, self.generator.dim))

        #Iterate the noise generation and whitening over ONE batch
        for i in range(self.generator.batch_size):
            noise_AET= self.generator.noise_td_AET(self.generator.dim, self.generator.dt, channels=self.generator.channels_dict[self.generator.TDI_channels])#["AE","AE","T"]
            x_test[i,:,:]= self.generator.noise_whiten_AET(noise_AET, self.generator.dt, channels=self.generator.channels_dict[self.generator.TDI_channels])
        
        #Reshape the batch of noise samples for input into the model 
        x_test= xp.reshape(x_test, (self.generator.batch_size, self.generator.dim, self.generator.n_channels))#.get()
        y_true= xp.zeros((self.generator.batch_size, self.generator.dim, self.generator.n_channels))
        #y_test= xp.reshape(y_test, (self.generator.batch_size, self.generator.dim, self.generator.n_channels)).get()

        #Convert input from xp array to TF tensor
        x_test= self.generator.cupy_to_tensor(x_test)

        #Make a prediction
        y_pred= self.model(x_test, training=False)#self.model.predict(x_test, verbose=0, batch_size= self.generator.batch_size)

        #Convert prediction from TF tensor to xp array
        y_pred= self.generator.tensor_to_cupy(y_pred)

        ''' Evaluate requires the input data, and the target data.
            For denoising, pure LISA noise --> Model ---> no noise at all'''
        

        '''Loss functions are supposed to be calculated with y_true and y_pred.
            In our case, y_true is denoised noise i.e. an array of zeros
            and y_pred is our model prediction. But we're using x instead of y_true!'''
        
        ''' Tensorflow's in-built MSE function is janky so let's do it by hand.
            The process is:
            1. Find the difference between y_true and y_pred
            2. Square that difference
            3. Take the mean of the squared difference over axis 1
            4. Sum the array of MSEs across the batch'''

        batch_loss= xp.sum(xp.mean((y_true-y_pred)**2, axis=1)).get()
        
        #batch_loss = self.model.evaluate(x_test, y_pred.numpy(), verbose=2)
        #batch_loss = MeanSquaredError(y_true, y_pred.numpy())
        #State and store the losses from these noise samples
        print("Noise loss: ", batch_loss)        
        self.losses.append(batch_loss)
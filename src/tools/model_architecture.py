import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal.windows import tukey


class ConvAE(nn.Module):
    def __init__(self):
      '''Defines all the layers of the
         model. This is not sequential!'''
      super(ConvAE, self).__init__()#what does this line mean?
      
      '''Quick idea to test. Why not just double or even 10x the padding? May help with
         getting rid of the artefacts.'''
      #Encoder convs
      self.conv1= nn.Conv1d(2, 32, 65, stride=8, padding=59)#29, 59
      self.conv2= nn.Conv1d(32, 64, 65, stride=8, groups=8, padding=59)#29, 59
      self.conv3= nn.Conv1d(64, 128, 65, stride=8, groups=8, padding=59)#originally 64->64 filters
      self.conv4= nn.Conv1d(128, 256, 65, stride=8, groups=8, padding=59)#originally 64->64 filters
      #Decoder convs
      self.deconv4= nn.ConvTranspose1d(256, 128, 65, stride=8, groups=8, padding=59, output_padding=5)#originally 64->64 filters
      self.deconv3= nn.ConvTranspose1d(128, 64, 65, stride=8, groups=8, padding=59, output_padding=5)#originally 64->64 filters
      self.deconv2= nn.ConvTranspose1d(64, 32, 65, stride=8, groups=8, padding=59,output_padding=4)#padding=29,output_padding=1 OR padding=59,output_padding=4
      self.deconv1= nn.ConvTranspose1d(32, 2, 65, stride=8, padding=59, output_padding=5)#padding=#29, output_padding=1 OR padding=59, output_padding=5
      
      #Encoder batch norm
      self.bn1= nn.BatchNorm1d(32)
      self.bn2= nn.BatchNorm1d(64)
      self.bn3= nn.BatchNorm1d(128)
      #Decoder batch norm
      self.debn4= nn.BatchNorm1d(128)
      self.debn3= nn.BatchNorm1d(64)
      self.debn2= nn.BatchNorm1d(32)

      #Encoder dropouts
      self.dropout_1= nn.Dropout(p=0.2)
      self.dropout_2= nn.Dropout(p=0.2)
      self.dropout_3= nn.Dropout(p=0.2)
      self.dropout_4= nn.Dropout(p=0.2)
      #Decoder dropouts
      self.dedropout_4= nn.Dropout(p=0.2)
      self.dedropout_3= nn.Dropout(p=0.2)
      self.dedropout_2= nn.Dropout(p=0.2)
      self.dedropout_1= nn.Dropout(p=0.2)


    def forward(self, x):# x: input data
      '''Defines the sequence
         of layers and activation functions that the input passes through,
         and returns the output of the model'''
      '''Some things to try:
         1. Experiment with fewer skip connections. 
            A lot of resnets don't actually have skip connections on every layer
            so there may be no need for so many. Some sources do it every two layers instead.
         
         2. ????
         3. ????'''
      enc_1= F.leaky_relu(self.conv1(x))
      # enc_1= self.bn1(enc_1)
      enc_1= self.dropout_1(enc_1)

      enc_2= F.leaky_relu(self.conv2(enc_1))
      # enc_2= self.bn2(enc_2)
      enc_2= self.dropout_2(enc_2)

      enc_3= F.leaky_relu(self.conv3(enc_2))
      # enc_3= self.bn3(enc_3)
      enc_3= self.dropout_3(enc_3)

      enc_4= F.leaky_relu(self.conv4(enc_3))
      enc_4= self.dropout_4(enc_4)

      dec_4= F.leaky_relu(self.deconv4(enc_4))
      # dec_4= self.debn4(dec_4)
      dec_4= self.dedropout_4(dec_4)

      dec_3= F.leaky_relu(self.deconv3(dec_4+enc_3))
      # dec_3= self.debn3(dec_3)
      dec_3= self.dedropout_3(dec_3)

      dec_2= F.leaky_relu(self.deconv2(dec_3+enc_2))
      # dec_2= self.debn2(dec_2)
      dec_2= self.dedropout_2(dec_2)

      dec_1= self.deconv1(dec_2+enc_1)

      return dec_1
    
    def padding_size(self, stride, L_out, L_in, dilation, kernel_size):
      '''Something strange about this function. It can sometimes give
         different answers for the same L_in and L_out when expressed
         as ints rather than 2**a and 2**b. Unclear why'''
      return np.ceil(0.5*(stride*(L_out-1)-L_in+dilation*(kernel_size-1)+1))#, dtype=int


class Dilated_ConvAE(nn.Module):
    def __init__(self):
      '''Defines all the layers of the
         model. This is not sequential!'''
      super(Dilated_ConvAE, self).__init__()#what does this line mean?
      '''PyTorch is a bit janky when it comes to padding in conv and tranpose conv layers.
         Zero-padding will have to be done by hand. Likely there are wrapper functions that
         people have written to do this. Find these!
         OR, maybe a different padding mode could be acceptable here?'''
      #Encoder convs
      self.conv1= nn.Conv1d(2, 32, 65, stride=8,  dilation=1, padding=29)#59
      self.conv2= nn.Conv1d(32, 64, 65, stride=8, dilation=2, groups=8, padding=61)#59
      self.conv3= nn.Conv1d(64, 128, 65, stride=8, dilation=4, groups=8, padding=125)#originally 64->64 filters
      self.conv4= nn.Conv1d(128, 256, 65, stride=8, dilation=8, groups=8, padding=253)#originally 64->64 filters
      #Decoder convs
      self.deconv4= nn.ConvTranspose1d(256, 128, 65, stride=8, dilation=8, groups=8, padding=253, output_padding=1)#originally 64->64 filters
      self.deconv3= nn.ConvTranspose1d(128, 64, 65, stride=8, dilation=4, groups=8, padding=125, output_padding=1)#originally 64->64 filters
      self.deconv2= nn.ConvTranspose1d(64, 32, 65, stride=8, dilation=2, groups=8, padding=61, output_padding=1)#padding=29,output_padding=1 OR padding=59,output_padding=4
      self.deconv1= nn.ConvTranspose1d(32, 2, 65, stride=8, dilation=1, padding=29, output_padding=1)#padding=#29, output_padding=1 OR padding=59, output_padding=5
      #Encoder dropouts
      self.dropout_1= nn.Dropout(p=0.2)
      self.dropout_2= nn.Dropout(p=0.2)
      self.dropout_3= nn.Dropout(p=0.2)
      self.dropout_4= nn.Dropout(p=0.2)
      #Decoder dropouts
      self.dedropout_4= nn.Dropout(p=0.2)
      self.dedropout_3= nn.Dropout(p=0.2)
      self.dedropout_2= nn.Dropout(p=0.2)
      self.dedropout_1= nn.Dropout(p=0.2)
    def forward(self, x):# x: input data
      '''Defines the sequence
         of layers and activation functions that the input passes through,
         and returns the output of the model'''
      enc_1= F.leaky_relu(self.conv1(x))
      enc_1= self.dropout_1(enc_1)
      enc_2= F.leaky_relu(self.conv2(enc_1))
      enc_2= self.dropout_2(enc_2)
      enc_3= F.leaky_relu(self.conv3(enc_2))
      enc_3= self.dropout_3(enc_3)
      enc_4= F.leaky_relu(self.conv4(enc_3))
      enc_4= self.dropout_4(enc_4)
      dec_4= F.leaky_relu(self.deconv4(enc_4))
      dec_4= self.dedropout_4(dec_4)
      dec_3= F.leaky_relu(self.deconv3(dec_4+enc_3))#
      dec_3= self.dedropout_3(dec_3)
      dec_2= F.leaky_relu(self.deconv2(dec_3+enc_2))#
      dec_2= self.dedropout_2(dec_2)
      dec_1= self.deconv1(dec_2+enc_1)#
      return dec_1
    def padding_size(self, stride, L_out, L_in, dilation, kernel_size):
      '''Something strange about this function. It can sometimes give
         different answers for the same L_in and L_out when expressed
         as ints rather than 2**a and 2**b. Unclear why'''
      return np.ceil(0.5*(stride*(L_out-1)-L_in+dilation*(kernel_size-1)+1))#, dtype=int

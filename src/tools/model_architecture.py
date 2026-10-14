import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
import numpy as np
from scipy.signal.windows import tukey

from src.tools.activations import *
from src.tools.layers import *
from src.tools.transforms import *


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
      '''
      Some hopefully easy ideas to try:
      1. Allow the time window to vary in position i.e. not only the first 4 months but a randomised 4-month window of an EMRI 
      2. Experiment with a multihead attention layer
      3. Experiment with Avi's pywavelet package - it is likely the key to capturing the entire EMRI in a reasonable dimensionality!
      '''
      #Encoder convs
      self.conv1= weight_norm(nn.Conv1d(2, 32, 65, stride=8,  dilation=1, padding=29, padding_mode="reflect"))#(dilation=1,padding=29,)
      self.smooth_conv2= weight_norm(nn.Conv1d(32, 32, 65,  groups=32, padding=32, padding_mode="reflect"))
      self.conv2= weight_norm(nn.Conv1d(32, 64, 65, stride=8, dilation=3, groups=8, padding=93, padding_mode="reflect"))#(dilation=2,padding=61)
      self.smooth_conv3= weight_norm(nn.Conv1d(64, 64, 65,  groups=64, padding=32, padding_mode="reflect"))
      self.conv3= weight_norm(nn.Conv1d(64, 128, 65, stride=8, dilation=5, groups=8, padding=157, padding_mode="reflect"))#(dilation=4, padding=125), (dilation=3,padding=93)
      self.smooth_conv4= weight_norm(nn.Conv1d(128, 128, 65,  groups=128, padding=32, padding_mode="reflect"))
      self.conv4= weight_norm(nn.Conv1d(128, 256, 65, stride=8, dilation=6, groups=8, padding=189, padding_mode="reflect"))#(dilation=8, padding=253), (dilation=4,padding=125)
      self.smooth_conv5= weight_norm(nn.Conv1d(256, 256, 65,  groups=256, padding=32, padding_mode="reflect"))
      self.conv5= weight_norm(nn.Conv1d(256, 512, 65, stride=8, dilation=7, groups=8, padding=221, padding_mode="reflect"))#(dilation=16, padding=509),(dilation=5,padding=157) 
      #Something weird: a self attention layer
      # self.layer_norm_1 = nn.LayerNorm(32)
      # self.multihead_attn = nn.MultiheadAttention(512, 8, batch_first=True)
      # self.layer_norm_2 = nn.LayerNorm(32)
      #Decoder convs
      '''
      We can try a repeating DR schedule rather than letting it get too large,
      e.g. [1,3,5,1,3,5]
      AND: let's do an fft of the residual itself: see if we really are reducing high-freq. noise
      '''
      #Using resize convolutions to reduce checkerboard artifacts
      self.upsample5 = nn.Upsample(scale_factor=8, mode='nearest')
      self.deconv5= weight_norm(nn.Conv1d(512, 256, 65, stride=1, dilation=6, groups=8, padding=189+3, padding_mode="reflect"))#dilation=16,padding=509+3
      self.upsample4 = nn.Upsample(scale_factor=8, mode='nearest')
      self.deconv4= weight_norm(nn.Conv1d(256, 128, 65, stride=1, dilation=5, groups=8, padding=157+3, padding_mode="reflect"))#dilation=8, padding=253+3
      self.upsample3 = nn.Upsample(scale_factor=8, mode='nearest')
      self.deconv3= weight_norm(nn.Conv1d(128, 64, 65, stride=1, dilation=3, groups=8, padding=93+3, padding_mode="reflect"))#dilation=4,padding=125+3
      self.upsample2 = nn.Upsample(scale_factor=8, mode='nearest')
      self.deconv2= weight_norm(nn.Conv1d(64, 32, 65, stride=1, dilation=1, groups=8, padding=29+3, padding_mode="reflect"))#dilation=2, padding=61+3
      self.upsample1 = nn.Upsample(scale_factor=8, mode='nearest')
      self.deconv1= nn.Conv1d(32, 2, 65, stride=1, dilation=1, padding=29+3, padding_mode="reflect")
      # self.deconv5= weight_norm(nn.ConvTranspose1d(512, 256, 65, stride=8, dilation=16, groups=8, padding=509, output_padding=1))
      # self.deconv4= weight_norm(nn.ConvTranspose1d(256, 128, 65, stride=8, dilation=8, groups=8, padding=253, output_padding=1))
      # self.deconv3= weight_norm(nn.ConvTranspose1d(128, 64, 65, stride=8, dilation=4, groups=8, padding=125, output_padding=1))
      # self.deconv2= weight_norm(nn.ConvTranspose1d(64, 32, 65, stride=8, dilation=2, groups=8, padding=61, output_padding=1))
      # self.deconv1= nn.ConvTranspose1d(32, 2, 65, stride=8, dilation=1, padding=29, output_padding=1)
      #Encoder dropouts
      self.dropout_1= nn.Dropout(p=0.2)
      self.dropout_2= nn.Dropout(p=0.2)
      self.dropout_3= nn.Dropout(p=0.2)
      self.dropout_4= nn.Dropout(p=0.2)
      self.dropout_5= nn.Dropout(p=0.2)
      #Decoder dropouts
      self.dedropout_5= nn.Dropout(p=0.2)
      self.dedropout_4= nn.Dropout(p=0.2)
      self.dedropout_3= nn.Dropout(p=0.2)
      self.dedropout_2= nn.Dropout(p=0.2)
      self.dedropout_1= nn.Dropout(p=0.2)
      #Channel shuffle
      self.channel_shuffle = nn.ChannelShuffle(8)
    def forward(self, x):# x: input data
      '''Defines the sequence
         of layers and activation functions that the input passes through,
         and returns the output of the model'''
      #Encoder
      enc_1= F.leaky_relu(self.conv1(x))
      enc_1= self.dropout_1(self.smooth_conv2(enc_1))
      enc_2= F.leaky_relu(self.conv2(enc_1))
      enc_2= self.dropout_2(self.smooth_conv3((enc_2)))
      enc_2= self.channel_shuffle(enc_2)
      enc_3= F.leaky_relu(self.conv3(enc_2))
      enc_3= self.dropout_3(self.smooth_conv4(enc_3))
      enc_3= self.channel_shuffle(enc_3)
      enc_4= F.leaky_relu(self.conv4(enc_3))
      enc_4= self.dropout_4(self.smooth_conv5(enc_4))
      enc_4= self.channel_shuffle(enc_4)
      enc_5= F.leaky_relu(self.conv5(enc_4))
      enc_5= self.dropout_5(enc_5)
      enc_5= self.channel_shuffle(enc_5)
      # #Reshape encodings for the self attention
      # embed = torch.swapaxes(enc_5, 1, 2)
      # embed = self.layer_norm_1(embed)
      # attn_output, _ = self.multihead_attn(embed, embed, embed)
      # # norm_attn_output = self.layer_norm_2(attn_output + embed)
      # # norm_attn_output = torch.swapaxes(norm_attn_output, 1, 2)
      # attn_output = torch.swapaxes(attn_output, 1, 2)
      # attn_output = F.leaky_relu(attn_output)
      #Decoder
      dec_5= self.upsample5(enc_5)
      dec_5= F.leaky_relu(self.deconv5(dec_5))#, attn_output
      dec_5= self.dedropout_5(dec_5)
      dec_5= self.channel_shuffle(dec_5)
      dec_4= self.upsample4(dec_5+enc_4)
      dec_4= F.leaky_relu(self.deconv4(dec_4))#+enc_4
      dec_4= self.dedropout_4(dec_4)
      dec_4= self.channel_shuffle(dec_4)
      dec_3= self.upsample3(dec_4+enc_3)
      dec_3= F.leaky_relu(self.deconv3(dec_3))#+enc_3
      dec_3= self.dedropout_3(dec_3)
      dec_3= self.channel_shuffle(dec_3)
      dec_2= self.upsample2(dec_3+enc_2)
      dec_2= F.leaky_relu(self.deconv2(dec_2))#+enc_2
      dec_2= self.dedropout_2(dec_2)
      dec_2= self.channel_shuffle(dec_2)
      dec_1= self.upsample1(dec_2+enc_1)
      dec_1= self.deconv1(dec_1)#+enc_1
      # dec_5= F.leaky_relu(self.deconv5(enc_5))#, attn_output
      # dec_5= self.dedropout_5(dec_5)
      # dec_5= self.channel_shuffle(dec_5)
      # dec_4= F.leaky_relu(self.deconv4(dec_5+enc_4))
      # dec_4= self.dedropout_4(dec_4)
      # dec_4= self.channel_shuffle(dec_4)
      # dec_3= F.leaky_relu(self.deconv3(dec_4+enc_3))#
      # dec_3= self.dedropout_3(dec_3)
      # dec_3= self.channel_shuffle(dec_3)
      # dec_2= F.leaky_relu(self.deconv2(dec_3+enc_2))#
      # dec_2= self.dedropout_2(dec_2)
      # dec_2= self.channel_shuffle(dec_2)
      # dec_1= self.deconv1(dec_2+enc_1)#
      return dec_1
    def padding_size(self, stride, L_out, L_in, dilation, kernel_size):
      '''Something strange about this function. It can sometimes give
         different answers for the same L_in and L_out when expressed
         as ints rather than 2**a and 2**b. Unclear why'''
      return np.ceil(0.5*(stride*(L_out-1)-L_in+dilation*(kernel_size-1)+1))#, dtype=int


##########################
#2D model architectures!
##########################
class ConvAE_2D(nn.Module):
    def __init__(self):
      '''Defines all the layers of the
         model. This is not sequential!'''
      super(ConvAE_2D, self).__init__()
      '''
      Desperately need to keep it simple. Currently two experiments: fit for mask, fit for signal directly.
      It may make training much easier if we pre-train on noiseless data!
      Do a run on noiseless EMRIs, use those model weights as the initial weights for the noisy training.
      Refinement head will stay off for the meantime as it's unclear whether it is actually improving the results
      '''
      '''
      To dos:
      2. Save additional metrics DURING training in the form of (rho_d - rho_opt) and overlap
      '''
      #Encoder convs
      self.conv1= GatedConv2dWithActivation(2, 32, 3, stride=(1,1), dilation=(1,1), padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, activation=torch.nn.Identity(), residual=False)#nn.Conv2d(2, 32, 3, stride=(1,1),  dilation=(1,1), padding=(1,1), bias=False)#(dilation=1,padding=29,)#, padding_mode="reflect"
      self.conv2= GatedConv2dWithActivation(32, 64, 3, stride=(2,2), dilation=(1,1), groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#(dilation=2,padding=61)#, padding_mode="reflect"
      self.conv3= GatedConv2dWithActivation(64, 64, 3, stride=(1,1), dilation=(1,1), groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#(dilation=4, padding=125), (dilation=3,padding=93)#, padding_mode="reflect"
      self.conv4= GatedConv2dWithActivation(64, 128, 3, stride=(2,2), dilation=(1,1), groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#(dilation=8, padding=253), (dilation=4,padding=125)#, padding_mode="reflect"
      self.conv5= GatedConv2dWithActivation(128, 128, 3, stride=(1,1), dilation=(1,1), groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#(dilation=16, padding=509),(dilation=5,padding=157)#, padding_mode="reflect"
      self.conv6= GatedConv2dWithActivation(128, 128, 3, stride=(2,2), dilation=(1,1), groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#(dilation=16, padding=509),(dilation=5,padding=157)#, padding_mode="reflect"
      self.conv7= GatedConv2dWithActivation(128, 128, 3, stride=(1,1), dilation=(1,1), groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#(dilation=16, padding=509),(dilation=5,padding=157) #, padding_mode="reflect"
      self.conv8= GatedConv2dWithActivation(128, 128, 3, stride=(2,2), dilation=(1,1), groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#(dilation=16, padding=509),(dilation=5,padding=157) #, padding_mode="reflect"
      #Bottleneck layers
      '''
      An idea: pyramid style dilated convs. Multiple parallel convs with their own dilations.
      Take the 128 features, convolve each using 4 groups of 32 filters dilated by [1,2,3,4].
      Then fuse with a final 1x1 conv to the original 128 features. Easy to code up!
      '''
      self.bottleneck_conv1= HybridASPP(128,128)#AsymmetricGatedConv2dWithActivation(128, 128, 3, stride=1, dilation=1, padding=1)
      self.CBAM_1 = CBAM(128, reduction=2)
      self.bottleneck_conv2= HybridASPP(128,128)#AsymmetricGatedConv2dWithActivation(128, 128, 3, stride=1, dilation=2, padding=2)
      self.CBAM_2 = CBAM(128, reduction=2)
      self.bottleneck_conv3= HybridASPP(128,128)#AsymmetricGatedConv2dWithActivation(128, 128, 3, stride=1, dilation=3, padding=3)
      self.CBAM_3 = CBAM(128, reduction=2)
      # self.bottleneck_conv4= AsymmetricGatedConv2dWithActivation(128, 128, 3, stride=1, dilation=4, padding=4)
      # self.bottleneck_conv5= AsymmetricGatedConv2dWithActivation(128, 128, 3, stride=1, dilation=5, padding=5)
      # self.bottleneck_conv6= AsymmetricGatedConv2dWithActivation(128, 128, 3, stride=1, dilation=4, padding=4)
      # self.bottleneck_conv7= AsymmetricGatedConv2dWithActivation(128, 128, 3, stride=1, dilation=3, padding=3)
      # self.bottleneck_conv8= AsymmetricGatedConv2dWithActivation(128, 128, 3, stride=1, dilation=2, padding=2)
      # self.bottleneck_conv9= AsymmetricGatedConv2dWithActivation(128, 128, 3, stride=1, dilation=1, padding=1)
      #Decoder convs
      '''Adding filters rather than concatenating!!!'''
      self.deconv8= GatedDeConv2dWithActivation(2, 128, 64, 3, stride=(1,1), dilation=1, groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#, padding_mode="reflect"
      self.deconv7= GatedDeConv2dWithActivation(1, 128, 64, 3, stride=(1,1), dilation=1, groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#dilation=16,padding=509+3#, padding_mode="reflect"
      self.deconv6= GatedDeConv2dWithActivation(2, 128, 64, 3, stride=(1,1), dilation=1, groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#, padding_mode="reflect"
      self.deconv5= GatedDeConv2dWithActivation(1, 128, 64, 3, stride=(1,1), dilation=1, groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#dilation=16,padding=509+3#, padding_mode="reflect"
      self.deconv4= GatedDeConv2dWithActivation(2, 128, 32, 3, stride=(1,1), dilation=1, groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#dilation=8, padding=253+3#, padding_mode="reflect"
      self.deconv3= GatedDeConv2dWithActivation(1, 64, 32, 3, stride=(1,1), dilation=1, groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#dilation=4,padding=125+3#, padding_mode="reflect"
      self.deconv2= GatedDeConv2dWithActivation(2, 64, 16, 3, stride=(1,1), dilation=1, groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#dilation=2, padding=61+3#, padding_mode="reflect"
      self.deconv1= GatedDeConv2dWithActivation(1, 32, 2, 1, stride=1, dilation=1, padding=(0,0), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#, padding_mode="reflect"

      # '''Concatenating filters rather than adding!!!'''
      # self.deconv8= GatedDeConv2dWithActivation(2, 128, 64, 3, stride=(1,1), dilation=1, groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#, padding_mode="reflect"
      # self.deconv7= GatedDeConv2dWithActivation(1, 128, 128, 3, stride=(1,1), dilation=1, groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#dilation=16,padding=509+3#, padding_mode="reflect"
      # self.deconv6= GatedDeConv2dWithActivation(2, 128, 64, 3, stride=(1,1), dilation=1, groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#, padding_mode="reflect"
      # self.deconv5= GatedDeConv2dWithActivation(1, 128, 128, 3, stride=(1,1), dilation=1, groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#dilation=16,padding=509+3#, padding_mode="reflect"
      # self.deconv4= GatedDeConv2dWithActivation(2, 128, 32, 3, stride=(1,1), dilation=1, groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#dilation=8, padding=253+3#, padding_mode="reflect"
      # self.deconv3= GatedDeConv2dWithActivation(1, 64, 64, 3, stride=(1,1), dilation=1, groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#dilation=4,padding=125+3#, padding_mode="reflect"
      # self.deconv2= GatedDeConv2dWithActivation(2, 64, 16, 3, stride=(1,1), dilation=1, groups=1, padding=(1,1), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#dilation=2, padding=61+3#, padding_mode="reflect"
      # self.deconv1= GatedDeConv2dWithActivation(1, 32, 2, 1, stride=1, dilation=1, padding=(0,0), bias=True, GN_num_groups=None, WS_conv=False, residual=True)#, padding_mode="reflect"
      '''
      Change the weight initialisation on the output conv layer! A poor choice can really slow down training.
      '''
      #Tune the weights on the feature conv filters on the output for tanh activation
      #Gain 1 for linear, 5/3 for tanh
      nn.init.xavier_normal_(self.deconv1.conv2d.conv2d.weight[:, :2, :, :], gain=1.0)#1.0  # feature
      #Initialise some scaling parameters for the bottleneck layers
      '''This is the ReZero technique.'''
      self.alpha_1 = nn.Parameter(torch.tensor(0.0))
      self.alpha_2 = nn.Parameter(torch.tensor(0.0))
      self.alpha_3 = nn.Parameter(torch.tensor(0.0))
      self.alpha_4 = nn.Parameter(torch.tensor(0.0))
      self.alpha_5 = nn.Parameter(torch.tensor(0.0))
      self.alpha_6 = nn.Parameter(torch.tensor(0.0))
      #Initialise some scaling parameters for the concatenative skips
      '''Notably, these are not zero since we don't want to completely kill the information
      flow at the start of training.'''
      self.skip_alpha_1 = nn.Parameter(torch.tensor(0.1))
      self.skip_alpha_2 = nn.Parameter(torch.tensor(0.1))
      self.skip_alpha_3 = nn.Parameter(torch.tensor(0.1))
      self.skip_alpha_4 = nn.Parameter(torch.tensor(0.1))

      self.skip_alpha_5 = nn.Parameter(torch.tensor(0.1))
      self.skip_alpha_6 = nn.Parameter(torch.tensor(0.1))
      self.skip_alpha_7 = nn.Parameter(torch.tensor(0.1))
      self.skip_alpha_8 = nn.Parameter(torch.tensor(0.1))
      #Refinement convs
      '''
      Something to note: gated convs have found to be really good in image inpainting.
      We could treat the refinement head as an inpainting problem, and use gated convs
      here too!
      '''
      self.refine_conv1 = GatedConv2dWithActivation(4, 32, 3, stride=(1,1), dilation=(1,1), groups=1, padding=(1,1), activation=torch.nn.Identity())#GatedConv2dWithActivation(2, 32, 3, stride=(1,1), dilation=(1,1), groups=1, padding=(1,1), activation=torch.nn.Identity())
      self.refine_conv2 = GatedConv2dWithActivation(32, 64, 3, stride=(1,1), dilation=(1,1), groups=1, padding=(1,1), activation=torch.nn.Identity())#GatedConv2dWithActivation(32, 64, 3, stride=(1,1), dilation=(1,1), groups=1, padding=(1,1))
      self.refine_conv3 = GatedConv2dWithActivation(64, 2, 1, stride=(1,1), dilation=(1,1), groups=1, padding=(0,0), activation=torch.nn.Identity())#GatedConv2dWithActivation(64, 2, 1, stride=(1,1), dilation=(1,1), groups=1, padding=(0,0))
      #Attention gates/ gated skips
      '''edited'''
      self.gated_skip_8 = SpectrogramGatedSkip(128, 64, out_chans=64)
      self.gated_skip_7= SpectrogramGatedSkip(128, 64, out_chans=64)
      self.gated_skip_6 = SpectrogramGatedSkip(128, 64, out_chans=64)
      self.gated_skip_5 = SpectrogramGatedSkip(128, 64, out_chans=64)
      self.gated_skip_4 = SpectrogramGatedSkip(64, 32, out_chans=32)
      self.gated_skip_3 = SpectrogramGatedSkip(64, 32, out_chans=32)
      self.gated_skip_2 = SpectrogramGatedSkip(32, 16, out_chans=16)

      # self.gated_skip_8 = SpectrogramGatedSkip(128, 64, out_chans=64)
      # self.gated_skip_6 = SpectrogramGatedSkip(128, 64, out_chans=64)
      # self.gated_skip_4 = SpectrogramGatedSkip(64, 32, out_chans=32)
      # self.gated_skip_2 = SpectrogramGatedSkip(32, 16, out_chans=16)
    
    def predict_coarse_mask(self, x):
      #Encoder
      '''Encoder is fully residual!'''
      enc_1= self.conv1(x)
      enc_2= self.conv2(enc_1)
      enc_3= self.conv3(enc_2)
      enc_4= self.conv4(enc_3)
      enc_5= self.conv5(enc_4)
      enc_6= self.conv6(enc_5)
      enc_7= self.conv7(enc_6)
      enc_8= self.conv8(enc_7)
      #Bottleneck
      bottleneck_1 = self.alpha_1 * self.CBAM_1(self.bottleneck_conv1(enc_8)) +  enc_8
      bottleneck_2 = self.alpha_2 * self.CBAM_2(self.bottleneck_conv2(bottleneck_1)) + bottleneck_1
      bottleneck_3 = self.alpha_3 * self.CBAM_3(self.bottleneck_conv3(bottleneck_2)) + bottleneck_2
      '''bottleneck_1 = self.alpha_1 * self.bottleneck_conv1(enc_8) +  enc_8
      bottleneck_1 = self.alpha_2 * self.CBAM_1(bottleneck_1) + bottleneck_1
      bottleneck_2 = self.alpha_3 * self.bottleneck_conv2(bottleneck_1) + bottleneck_1
      bottleneck_2 = self.alpha_4 * self.CBAM_2(bottleneck_2) + bottleneck_2
      bottleneck_3 = self.alpha_5 * self.bottleneck_conv3(bottleneck_2) + bottleneck_2
      bottleneck_3 = self.alpha_6 * self.CBAM_3(bottleneck_3) + bottleneck_3'''
      #bottleneck_3 = enc_8
      #Decoder
      '''Decoder is fully residual!'''
      dec_8= self.deconv8(bottleneck_3)
      dec_8= torch.cat((dec_8, self.skip_alpha_8 * self.gated_skip_8(enc_7, dec_8)), dim=1)#skip

      dec_7= self.deconv7(dec_8)
      dec_7= torch.cat((dec_7, self.skip_alpha_7 * self.gated_skip_7(enc_6, dec_7)), dim=1)

      dec_6= self.deconv6(dec_7)
      dec_6= torch.cat((dec_6, self.skip_alpha_6 * self.gated_skip_6(enc_5, dec_6)), dim=1)#skip

      dec_5= self.deconv5(dec_6)
      dec_5= torch.cat((dec_5, self.skip_alpha_5 * self.gated_skip_5(enc_4, dec_5)), dim=1)

      dec_4= self.deconv4(dec_5)
      dec_4= torch.cat((dec_4, self.skip_alpha_4 * self.gated_skip_4(enc_3, dec_4)), dim=1)

      dec_3= self.deconv3(dec_4)
      dec_3= torch.cat((dec_3, self.skip_alpha_3 * self.gated_skip_3(enc_2, dec_3)), dim=1)

      dec_2= self.deconv2(dec_3)
      dec_2= torch.cat((dec_2, self.skip_alpha_2 * self.gated_skip_2(enc_1, dec_2)), dim=1)#skip

      coarse_mask= self.deconv1(dec_2)

      return coarse_mask
    def predict_refined_signal(self, x, coarse_mask):
      coarse_masked = F.tanh(x * coarse_mask)
      coarse_masked_and_input = torch.cat((coarse_masked, x), dim=1)#
      '''
      Interesting suggestion from ChatGPT. Concatenate the original input too! Let the conv see
      both the masked data and the original clean data.
      '''
      #Above layer sort of acts as a pre-activation!
      refine_1 = self.refine_conv1(coarse_masked_and_input)
      refine_2 = self.refine_conv2(F.silu(refine_1, inplace=True))
      refine_3 = self.refine_conv3(F.silu(refine_2, inplace=True))#refine_2
      denoised = F.tanh(refine_3 + coarse_masked)#Effectively F(x) + x
      return denoised
    
    def forward(self, x):# x: input data
      '''Defines the sequence
         of layers and activation functions that the input passes through,
         and returns the output of the model'''
      #Predict mask
      '''Remember, this mask has been preprocessed!'''
      coarse_mask = self.predict_coarse_mask(x)
      '''
      Residual approach: either do
      1. M' = y + sign(y)ln(2)
      2. M' = sign(X)y + sign(y)X + sign(y)sign(x)ln(2)
      Remember y should be normalised so make the other terms fit
      '''
      #Un-preprocess mask prior to applying it on input data
      '''asinh scaling has changed'''
      max_abs= 80.0#45.0
      max_abs_tensor= 53.0#106.0#
      inverse=True
      #Also unpreprocess the input too
      asinh_scale= 1e23#1e46#
      scaling_factor= 1e28#1e14
      #Apply residual approach
      '''Use the simpler residual approach'''
      '''NOTE: maybe remove the division by max abs since it otherwise becomes very small
      Consider that +-log(2) is ~+-0.69 which is roughly at the centre of each peak'''
      # coarse_mask = coarse_mask + torch.sign(coarse_mask)*torch.log(torch.tensor(2.0))#(1/max_abs)*
      # coarse_mask = torch.sign(x)*coarse_mask + (torch.sign(coarse_mask)*x + torch.sign(coarse_mask)*torch.sign(x)*torch.log(torch.tensor(2)))/max_abs
      # coarse_mask = torch.sign(x)*coarse_mask + torch.sign(x)*torch.sign(coarse_mask)*torch.log(torch.tensor(2*scaling_factor)) - torch.sign(coarse_mask)*max_abs_tensor*x#/max_abs
      #Mask targets are mostly bimodal about +-1 so use a tanh activation
      coarse_mask = torch.clamp(coarse_mask, min=-1, max=1)#torch.tanh(coarse_mask)#torch.clamp(coarse_mask, min=-1, max=1)#torch.tanh(coarse_mask)##1.3 * torch.tanh(coarse_mask)
      #Denoised is UN-processed!!!
      denoised = preprocess_signal(x, 
                              asinh_scale=asinh_scale, 
                            #   asinh_shift=asinh_shift, 
                              max_absolute=max_abs_tensor,
                              inverse = inverse
                              ) * preprocess_mask(coarse_mask, 
                                        max_abs=max_abs,
                                        scaling_factor=scaling_factor,
                                #  power_factor=power_factor, 
                                 inverse=inverse
                                 )
      '''
      Another step for stability. Because we have divided the data by the maximum absolute,
      we can bound the output with the tanh function
      '''
      #Apply mask on input
      #denoised = coarse_mask * x#torch.concatenate((x,x), dim=1)#self.predict_refined_signal(x, coarse_mask)
      return coarse_mask, denoised
      #
      # #Encoder
      # enc_1= F.leaky_relu(self.conv1(x))
      # enc_1= self.dropout_1(enc_1)
      # enc_2= F.leaky_relu(self.conv2(enc_1))
      # enc_2= self.dropout_2((enc_2))
      # # enc_2= self.channel_shuffle(enc_2)
      # enc_3= F.leaky_relu(self.conv3(enc_2))
      # enc_3= self.dropout_3((enc_3))
      # # enc_3= self.channel_shuffle(enc_3)
      # enc_4= F.leaky_relu(self.conv4(enc_3))
      # enc_4= self.dropout_4((enc_4))
      # # enc_4= self.channel_shuffle(enc_4)
      # enc_5= F.leaky_relu(self.conv5(enc_4))
      # enc_5= self.dropout_5(enc_5)
      # # enc_5= self.channel_shuffle(enc_5)
      # #Decoder
      # # dec_5= self.upsample5(enc_5)
      # dec_5= F.leaky_relu(self.deconv5(enc_5))+enc_4#dec5
      # dec_5= self.dedropout_5(dec_5)
      # # dec_5= self.channel_shuffle(dec_5)
      # dec_4= self.upsample4(dec_5)
      # dec_4= F.leaky_relu(self.deconv4(dec_4))+enc_3
      # dec_4= self.dedropout_4(dec_4)
      # # dec_4= self.channel_shuffle(dec_4)
      # # dec_3= self.upsample3(dec_4)
      # dec_3= F.leaky_relu(self.deconv3(dec_4))+enc_2
      # dec_3= self.dedropout_3(dec_3)
      # # dec_3= self.channel_shuffle(dec_3)
      # dec_2= self.upsample2(dec_3)
      # dec_2= F.leaky_relu(self.deconv2(dec_2))+enc_1
      # dec_2= self.dedropout_2(dec_2)
      # # dec_2= self.channel_shuffle(dec_2)
      # # dec_1= self.upsample1(dec_2+enc_2)
      # dec_1= self.deconv1(dec_2)#+enc_1
      # return dec_1
    def padding_size(self, stride, L_out, L_in, dilation, kernel_size):
      '''Something strange about this function. It can sometimes give
         different answers for the same L_in and L_out when expressed
         as ints rather than 2**a and 2**b. Unclear why'''
      return np.ceil(0.5*(stride*(L_out-1)-L_in+dilation*(kernel_size-1)+1))#, dtype=int

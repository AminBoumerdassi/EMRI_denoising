import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

'''
Borrowed and adapted things from:
1. https://github.com/avalonstrel/GatedConvolution_pytorch/blob/master/models/networks.py
2. https://github.com/joe-siyuan-qiao/WeightStandardization
3. https://github.com/AhmedAShaheen/fully_gated_DAE
'''

def standardise_weights_2D(weight):
    weight = weight - weight.mean(dim=(1, 2, 3), keepdim=True)
    var = weight.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
    weight = weight / torch.sqrt(var + 1e-5)
    return weight


class Conv2d_WS(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_WS, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = standardise_weights_2D(self.weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def init_weights(net, init_type='normal', gain=0.02):
    from torch.nn import init
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)

class GatedConv2dWithActivation(torch.nn.Module):
    """
    PRE-ACTIVATION style gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                  stride=1, padding=0, dilation=1, groups=1,
                    bias=True, GN_num_groups=None, WS_conv=False,
                      activation=torch.nn.SiLU(inplace=False),
                      residual=False):
        super(GatedConv2dWithActivation, self).__init__()
        self.activation = activation
        self.GN_num_groups = GN_num_groups
        self.WS_conv = WS_conv
        self.residual = residual
        '''Simplified gated conv which merges the feature extraction and gating into one'''
        #Initialise type of conv
        conv_func = Conv2d_WS if self.WS_conv else torch.nn.Conv2d
        #Initialise conv layers
        self.conv2d = conv_func(in_channels, 2 * out_channels, kernel_size, stride, padding, dilation, groups, bias=True)
        self.res_1x1_conv2d = conv_func(in_channels, out_channels, 1, stride, 0, dilation, groups, bias=True)
        #Initialise norm layer
        self.group_norm2d = torch.nn.GroupNorm(self.GN_num_groups, in_channels) if self.GN_num_groups is not None else torch.nn.Identity()
        #Initialise sigmoid function for gating
        self.sigmoid = torch.nn.Sigmoid()
        #Initialise different init weight schemes on shared conv and skip gate
        nn.init.kaiming_normal_(self.conv2d.weight[:, :out_channels, :, :], mode='fan_out', nonlinearity='relu')  # feature
        nn.init.normal_(self.conv2d.weight[:, out_channels:, :, :], mean=0.0, std=0.01)  # gate
        #Initialise different biases on shared conv
        with torch.no_grad():
            self.conv2d.bias[:out_channels] = 0.0#feature bias
            self.conv2d.bias[out_channels:] = 1.0#apparently better for gating in deep nets
    def gated(self, mask):
        return self.sigmoid(mask)
    def forward(self, input):
        # normalise
        x = self.group_norm2d(input)
        # activation
        x = self.activation(x)
        # shared feature and mask conv
        features, mask = self.conv2d(x).chunk(2, dim=1)
        # gated conv
        gated = torch.nn.Identity()(features) * self.gated(mask)
        # residual
        if self.residual:
            res = self.res_1x1_conv2d(input) if input.shape != gated.shape else input
            gated = gated + res
        return gated

class GatedDeConv2dWithActivation(torch.nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, GN_num_groups=None, WS_conv=False, activation=torch.nn.SiLU(inplace=True), residual=False):
        super(GatedDeConv2dWithActivation, self).__init__()
        self.conv2d = GatedConv2dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, GN_num_groups, WS_conv, activation, residual)
        self.scale_factor = scale_factor
        self.upscale = torch.nn.Upsample(scale_factor=self.scale_factor, mode='nearest')

    def forward(self, input):
        #print(input.size())
        x = self.upscale(input)#F.interpolate(input, scale_factor=2)
        return self.conv2d(x)
    
class AsymmetricGatedConv2dWithActivation(torch.nn.Module):
    """
    Asymmetric kernel gated Convlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """
    def __init__(self, in_channels, inter_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, GN_num_groups=None, WS_conv=False, activation=torch.nn.SiLU()):
        super(AsymmetricGatedConv2dWithActivation, self).__init__()
        self.conv2d_in = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=True)
        self.conv2d_time = GatedConv2dWithActivation(inter_channels, inter_channels, (kernel_size, 1), (stride,1), (padding,0), (dilation,1), groups, bias, GN_num_groups, WS_conv, activation)
        self.conv2d_freq = GatedConv2dWithActivation(inter_channels, inter_channels, (1, kernel_size), (1,stride), (0,padding), (1,dilation), groups, bias, GN_num_groups, WS_conv, activation)
        self.conv2d_out = nn.Conv2d(inter_channels, out_channels, kernel_size=1, bias=True)
        
    def forward(self, input):
        out = self.conv2d_in(input)
        out = self.conv2d_time(out)
        out = self.conv2d_freq(out)
        out = self.conv2d_out(out)
        return out

class GatedSkip(torch.nn.Module):
    """
    Gated U-Net skip connections

    Args:
        in_enc: channels of encoder feature map
        in_dec: channels of decoder feature map (gating signal)
        residual: if True, use residual gating (x * gate + x)
    """
    def __init__(
        self,
        in_enc,
        in_dec,
        out_chans,
        residual=False
    ):
        super().__init__()

        if inter_channels is None:
            inter_channels = max(1, in_enc // 2)

        if out_chans is None:
            out_chans = in_enc

        def conv1x1(in_ch, out_ch):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)]
            return nn.Sequential(*layers)

        # Transform decoder feature (gating signal)
        self.gating = nn.Sequential(
            conv1x1(in_dec + in_enc, in_dec + in_enc),
            nn.Sigmoid()
        )
        
        # Final conv to get the desired no. of channels
        self.xi = conv1x1(in_enc, out_chans)

        self.relu = torch.nn.SiLU(inplace=True)
        self.residual = residual

        self._init_weights()

    def _init_weights(self):
        # Initialize psi bias so gates start "open"
        for m in self.gating.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, 1.0)

    def forward(self, x_enc, x_dec):
        """
        x_enc: encoder feature (B, C_e, H, W)
        x_dec: decoder feature (B, C_d, H', W')
        """
        
        #Gating on concatenated enc and dec features
        gate = self.gating(self.relu(torch.cat((x_enc, x_dec), dim=1)))

        # Apply gating
        out = x_enc * gate

        # Conv to the desired no. of chans
        out = self.xi(out)

        if self.residual:
            out = out + x_enc

        return out
    

#With thanks to Professor G.P.T
class SpectrogramGatedSkip(nn.Module):
    def __init__(self, in_enc, in_dec, out_chans=None, inplace=False):
        super().__init__()

        inter = max(8, in_enc // 2)
        out_chans = out_chans or in_enc

        self.enc_proj = nn.Sequential(
            torch.nn.SiLU(inplace=inplace),
            nn.Conv2d(in_enc, inter, 1),
        )
        
        self.dec_proj = nn.Sequential(
            torch.nn.SiLU(inplace=inplace),
            nn.Conv2d(in_dec, inter, 1),
        )

        self.fuse = nn.Sequential(
            torch.nn.SiLU(inplace=inplace),
            nn.Conv2d(inter * 2, inter, 3, padding=1),
        )

        # Channel-aware gating
        self.gate = nn.Sequential(
            torch.nn.SiLU(inplace=inplace),
            nn.Conv2d(inter, in_enc, 1),
            nn.Sigmoid(),
        )

        self.out_conv = nn.Conv2d(in_enc, out_chans, 1)

    def forward(self, x_enc, x_dec):
        if x_enc.shape[2:] != x_dec.shape[2:]:
            x_dec = F.interpolate(x_dec, size=x_enc.shape[2:], mode='bilinear', align_corners=False)

        e = self.enc_proj(x_enc)
        d = self.dec_proj(x_dec)

        f = self.fuse(torch.cat([e, d], dim=1))

        gate = self.gate(f)

        out = x_enc * gate
        out = self.out_conv(out)

        return out

#With thanks to Professor G.P.T
class HybridASPP(nn.Module):
    '''Hybrid-dilated pyramid of convs with stacked anisotropic kernels'''
    def __init__(self, in_ch, out_ch):
        super().__init__()

        out_branch_ch = out_ch // 4  # output channel of each branch
        inter_chan = out_ch // 2 #output channels of intermediate convs within each conv branch

        #Branch for 1x1 conv
        self.b1 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_ch, out_branch_ch, kernel_size=1, bias=True),
            )#AsymmetricGatedConv2dWithActivation(in_ch, inter_chan, out_branch_ch, 3, dilation=1, padding=1)

        # --- Anisotropic branches ---
        self.b2 = AsymmetricGatedConv2dWithActivation(in_ch, inter_chan, out_branch_ch, 3, dilation=2, padding=2)
        self.b3 = AsymmetricGatedConv2dWithActivation(in_ch, inter_chan, out_branch_ch, 3, dilation=4, padding=4)
        self.b4 = AsymmetricGatedConv2dWithActivation(in_ch, inter_chan, out_branch_ch, 3, dilation=8, padding=8)

        #Pooling branch
        self.b5 = nn.Sequential(
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),   # GAP
            nn.Conv2d(in_ch, inter_chan, 1),
        )

        # --- Fusion ---
        self.fuse = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(out_ch + inter_chan, out_ch, kernel_size=1),
        )

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)

        b5 = self.b5(x)
        #interpolate the pooling branch to match the dims of the other branches
        b5 = F.interpolate(b5, size=x.shape[2:], mode='bilinear', align_corners=False)

        out = torch.cat([b1, b2, b3, b4, b5], dim=1)
        out = self.fuse(out)

        return out

#With thanks to Professor G.P.T
class CBAM(nn.Module):
    def __init__(self, channels, reduction=2):
        super().__init__()

        # Channel attention (uses global pooling!)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, 1)
        )

        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        #No pre-activation

        # ----- Channel attention -----
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)

        ca = self.mlp(avg_pool) + self.mlp(max_pool)
        ca = torch.sigmoid(ca)
        x = x * ca

        # ----- Spatial attention -----
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)

        sa = torch.cat([avg, max_], dim=1)
        sa = torch.sigmoid(self.spatial(sa))

        x = x * sa
        return x
import numpy as np
import torch
from torch import nn

#From LapSRN-PyTorch github repo
class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    def forward(self, pred: torch.Tensor, target: torch.Tensor, region_weighting = None, sample_weighting = None) -> torch.Tensor:
        loss = torch.sqrt(torch.pow((pred - target), 2) + self.eps)
        if region_weighting is not None:
            loss = region_weighting * loss
        if sample_weighting is not None:
            loss = sample_weighting * loss
        loss = torch.mean(loss)
        # '''Here, we are reducing by summing rather than averaging!!!'''
        # loss = torch.sum(torch.sqrt(torch.pow((pred - target), 2) + self.eps), dim=(-1,-2,-3))
        # loss = loss.mean()
        return loss


class OverlapLoss(torch.nn.Module):
    """
    Wavelet domain mismatch as a loss function.
    NOTE: overlap and mismatch are bounded between [-1,1] NOT [0,1]
    because you can have anticorrelated signals producing negative overlaps.
    """

    def __init__(self, eps=1e-8):#,  reduction="mean"
        super(OverlapLoss, self).__init__()
        self.eps = eps
        # self.reduction = reduction

    def forward(self, input, target):
        input_dot_target = torch.nansum(input*target, dim=(-2,-1))
        target_dot_target = torch.nansum(target*target, dim=(-2,-1))
        target_dot_input = torch.nansum(input*input, dim=(-2,-1))
        match = input_dot_target * torch.rsqrt(target_dot_target * target_dot_input + self.eps) #/ ( * ) )**0.5#
        #log the mismatch for easier optimisation
        #log_match = torch.log((1 - match)**2 + 1 + self.eps)
        log_match = 1 - match#(1 - match)**2#-torch.log(1 - torch.abs(match)**2)
        losses = log_match.mean()#apply_reduction(losses, self.reduction)
        return losses
    
class DetSNRLoss(torch.nn.Module):
    """
    Detected SNR as a loss function.
    Charbonnier loss between the detected SNR and optimal SNR
    """

    def __init__(self, eps=1e-8):#,  reduction="mean"
        super(DetSNRLoss, self).__init__()
        self.eps = eps
        # self.reduction = reduction

    def forward(self, input, target):
        opt_SNR = torch.sqrt(torch.nansum((target * target), dim=(-2,-1)))
        det_SNR = torch.nansum((input * target), dim=(-2,-1)) / (opt_SNR + self.eps)
        losses = torch.mean(torch.sqrt(torch.pow(det_SNR - opt_SNR, 2) + self.eps))
        return losses

class CorrelationLoss(torch.nn.Module):
    """
    pearson correlation as a loss function.
    Bounded between -1 and 1.
    """

    def __init__(self, eps=1e-8):#,  reduction="mean"
        super(CorrelationLoss, self).__init__()
        self.eps = eps
        # self.reduction = reduction
    
    def pearson_correlation(self, x, y):
        mean_x = torch.mean(x, dim=(-2,-1), keepdim=True)
        mean_y = torch.mean(y, dim=(-2,-1), keepdim=True)
        xm = x - mean_x
        ym = y - mean_y
        # Use torch.sum for the numerator (covariance)
        cov = torch.nansum(xm * ym, dim=(-2,-1))
        # Use torch.norm for stability and avoiding potential NaNs with sqrt(sum(...))
        # Add a small epsilon for numerical stability in the denominator
        norm_xm = torch.linalg.matrix_norm(xm, dim=(-2,-1))
        norm_ym = torch.linalg.matrix_norm(ym, dim=(-2,-1))
        denominator = norm_xm * norm_ym + self.eps
        pcc = cov / denominator
        return pcc

    def forward(self, input, target):
        losses = 1 - self.pearson_correlation(input, target)
        losses = torch.mean(losses)
        return losses

#Borrowed from https://github.com/mmany/pytorch-GDL/blob/main/custom_loss_functions.py
class GradientDifferenceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=1.0):
        super(GradientDifferenceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, targets):

        # gradient_diff = (inputs.diff(dim=-2, prepend=torch.tensor(0.0))-targets.diff(dim=-2, prepend=torch.tensor(0.0))).pow(2) + (inputs.diff(dim=-1, prepend=torch.tensor(0.0))-targets.diff(dim=-1, prepend=torch.tensor(0.0))).pow(2)
        term_1 = torch.diff(inputs, dim=-2).abs() - torch.diff(targets, dim=-2).abs() #(inputs.diff(dim=-2)-targets.diff(dim=-2)).pow(2)
        term_2 = torch.diff(inputs, dim=-1).abs() - torch.diff(targets, dim=-1).abs()  #(inputs.diff(dim=-1)-targets.diff(dim=-1)).pow(2)
        gradient_diff = torch.abs(term_1[:,:,:,:-1])**self.alpha + torch.abs(term_2[:,:,:-1,:])**self.alpha #term_1[:,:,:,:-1] + term_2[:,:,:-1,:]
        loss_gdl = torch.mean(gradient_diff)#gradient_diff.sum()/inputs.numel()#Maybe should be averaged just over batch size?

        return loss_gdl

# import math
# import torch
# from torch import autograd as autograd
# from torch import nn as nn
# from torch.nn import functional as F

# # # from basicsr.archs.vgg_arch import VGGFeatureExtractor
# # from basicsr.utils.registry import LOSS_REGISTRY
# # from .loss_util import weighted_loss

# # _reduction_modes = ['none', 'mean', 'sum']

# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)

# class CharbonnierLoss(nn.Module):
#     """Charbonnier loss (one variant of Robust L1Loss, a differentiable

#     variant of L1Loss).



#     Described in "Deep Laplacian Pyramid Networks for Fast and Accurate

#         Super-Resolution".



#     Args:

#         loss_weight (float): Loss weight for L1 loss. Default: 1.0.

#         reduction (str): Specifies the reduction to apply to the output.

#             Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.

#         eps (float): A value used to control the curvature near zero. Default: 1e-12.

#     """

#     def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
#         super(CharbonnierLoss, self).__init__()
#         if reduction not in ['none', 'mean', 'sum']:
#             raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

#         self.loss_weight = loss_weight
#         self.reduction = reduction
#         self.eps = eps

#     def forward(self, pred, target,  **kwargs):#weight=None,
#         """

#         Args:

#             pred (Tensor): of shape (N, C, H, W). Predicted tensor.

#             target (Tensor): of shape (N, C, H, W). Ground truth tensor.

#             weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.

#         """
#         return self.loss_weight * charbonnier_loss(pred, target, eps=self.eps, reduction=self.reduction)#weight,

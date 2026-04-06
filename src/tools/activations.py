import numpy as np
import torch
from torch import nn

class AsinhActivation(nn.Module):
    def forward(self, x):
        return torch.asinh(x)

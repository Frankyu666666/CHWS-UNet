import torch.nn as nn
import torch.nn.functional as F
import torch
# from .EachModule import HSAttention, CSAttention, WSAttention
from .ECAEModule import HSAttention, CSAttention, WSAttention

class CHWS(nn.Module):
    def __init__(self, kernel_size=7, in_planes=None, ratio=8, mac_pattern=0, mic_pattern=0):
        super(CHWS, self).__init__()
        self.cs = CSAttention(in_planes=in_planes[0], ratio=ratio, kernel_size=kernel_size, pattern=mic_pattern)
        self.hs = HSAttention(in_planes=in_planes[1], ratio=ratio, kernel_size=kernel_size, pattern=mic_pattern)
        self.ws = WSAttention(in_planes=in_planes[2], ratio=ratio, kernel_size=kernel_size, pattern=mic_pattern)
        self.bn = nn.BatchNorm2d(in_planes[0])
        self.relu = nn.ReLU()
        self.pattern = mac_pattern
        # self.mic_pattern = mic_pattern

    def forward(self, x):
        if self.pattern == 0: # All summed up
            x0 = self.cs(x)
            x1 = self.hs(x)
            x2 = self.ws(x)
            out = (x0 + x1 + x2)/3.0
        elif self.pattern == 1: # All in a row
            x0 = self.cs(x)
            x1 = self.hs(x0)
            out = self.ws(x1)
        elif self.pattern == 2:
            x0 = self.cs(x)
            x1 = self.hs(x0)
            x2 = self.ws(x0)
            out = 0.5*(x1 + x2)
        elif self.pattern == 3:
            x0 = self.cs(x)
            x1 = self.hs(x0)
            x2 = self.ws(x0)
            out = (x0 + x1 + x2)/3.0
        out = self.relu(out)
        out = self.bn(out)
        return out


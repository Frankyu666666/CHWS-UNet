import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from model.UNet_LCAM import ChannelAttention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class HSAttention(nn.Module):
    def __init__(self, in_planes=None, kernel_size=7, ratio=8, pattern=0):
        super(HSAttention, self).__init__()
        self.ca = ChannelAttention(in_planes=in_planes, ratio=ratio, pattern=3)
        self.sa = SpatialAttention(kernel_size)
        self.bn = nn.BatchNorm2d(in_planes)
        self.p = pattern
        self.relu = nn.ReLU()

    def forward(self, x):
        B, C, W, H = x.shape
        y = torch.reshape(x, (B, H, W, C))
        if self.p == 0:
            y = y * self.ca(y)
            y = y * self.sa(y)
        elif self.p == 1:
            y = y * self.sa(y)
            y = y * self.ca(y)
        else:
            y0 = y * self.ca(y)
            y1 = y * self.sa(y)
            y = 0.5*(y0 + y1)
        y = self.bn(y)
        out = self.relu(y)
        out = torch.reshape(out, (B, C, W, H))
        return out

class CSAttention(nn.Module):
    def __init__(self, in_planes=None, kernel_size=7, ratio=8, pattern=0):
        super(CSAttention, self).__init__()
        self.ca = ChannelAttention(in_planes=in_planes, ratio=ratio, pattern=3)
        self.sa = SpatialAttention(kernel_size)
        self.bn = nn.BatchNorm2d(in_planes)
        self.p = pattern
        self.relu = nn.ReLU()

    def forward(self, x):
        y = x
        if self.p == 0:
            y = y * self.ca(y)
            y = y * self.sa(y)
        elif self.p == 1:
            y = y * self.sa(y)
            y = y * self.ca(y)
        else:
            y0 = y * self.ca(y)
            y1 = y * self.sa(y)
            y = 0.5*(y0 + y1)
        y = self.bn(y)
        out = self.relu(y)
        return out

class WSAttention(nn.Module):
    def __init__(self, in_planes=None, kernel_size=7, ratio=8, pattern=0):
        super(WSAttention, self).__init__()
        self.ca = ChannelAttention(in_planes=in_planes, ratio=ratio, pattern=3)
        self.sa = SpatialAttention(kernel_size)
        self.bn = nn.BatchNorm2d(in_planes)
        self.p = pattern
        self.relu = nn.ReLU()

    def forward(self, x):
        B, C, W, H = x.shape
        y = torch.reshape(x, (B, W, C, H))
        if self.p == 0:
            y = y * self.ca(y)
            y = y * self.sa(y)
        elif self.p == 1:
            y = y * self.sa(y)
            y = y * self.ca(y)
        else:
            y0 = y * self.ca(y)
            y1 = y * self.sa(y)
            y = 0.5*(y0 + y1)
        y = self.bn(y)
        out = self.relu(y)
        out = torch.reshape(out, (B, C, W, H))
        return out


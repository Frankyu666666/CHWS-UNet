import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from model.UNet_LCAM import ChannelAttention

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=8, gamma=2, b=1):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.in_planes = in_planes
#         # self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
#         #                         nn.ReLU(),
#         #                         nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
#         # self.sigmoid = nn.Sigmoid()
#         kernel_size = int(abs((math.log(self.in_planes, 2) + b) / gamma))
#         kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
#         # self.channel_shuffle = ChannelShuffle(groups=groups)
#         self.con1 = nn.Conv1d(1,
#                               1,
#                               kernel_size=kernel_size,
#                               padding=(kernel_size - 1) // 2,
#                               bias=False)
#         self.act1 = nn.Sigmoid()
#
#     def forward(self, x):
#         out1 = self.avg_pool(x) + self.max_pool(x)
#         output = out1.squeeze(-1).transpose(-1, -2)
#         output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
#         output = self.act1(output)
#
#         # avg_out = self.fc(self.avg_pool(x))
#         # max_out = self.fc(self.max_pool(x))
#         # out = avg_out + max_out
#         # output = torch.multiply(x, output)
#         return output

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


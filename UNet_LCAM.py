# from model.Internal_Arrange.CHWSModule import CHWS
# from model.Internal_Arrange.EachModule import HSAttention, CSAttention, WSAttention
# from model.Internal_Arrange.ECAEModule import HSAttention, CSAttention, WSAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import numpy as np
# from model.Internal_Arrange.ECAEModule import ChannelAttention
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8, gamma=2, b=1, pattern=3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.in_planes = in_planes
        # self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
        #                         nn.ReLU(),
        #                         nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        # self.sigmoid = nn.Sigmoid()
        kernel_size = int(abs((math.log(self.in_planes, 2) + b) / gamma))
        kernel_size = np.max([kernel_size, 3])
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # self.channel_shuffle = ChannelShuffle(groups=groups)
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()
        self.pattern = pattern

    def forward(self, x):
        if self.pattern == 0:
            out1 = self.avg_pool(x) + self.max_pool(x)
        elif self.pattern == 1:
            out1 = self.avg_pool(x)
        elif self.pattern == 2:
            out1 = self.max_pool(x)
        else:
            output1 = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
            output1 = self.con1(output1).transpose(-1, -2).unsqueeze(-1)

            output2 = self.max_pool(x).squeeze(-1).transpose(-1, -2)
            output2 = self.con1(output2).transpose(-1, -2).unsqueeze(-1)
            out1 = output1 + output2

        if self.pattern != 3:
            out1 = out1.squeeze(-1).transpose(-1, -2)
            out1 = self.con1(out1).transpose(-1, -2).unsqueeze(-1)

        output = self.act1(out1)
        # output = self.act1(out1)
        return output

class Model(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, shortcut=[0, 0, 0, 0], pattern=0):
        super(Model, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)
        self.shortcut = shortcut
        self.pattern = pattern

        for i in range(0, len(shortcut)):
            if shortcut[i]:
                setattr(self, f"tp{i + 1}", ChannelAttention(in_planes=64*pow(2, i), pattern=self.pattern))

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        if self.shortcut[3]:
            x4 = x4*self.tp4(x4)

        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        if self.shortcut[2]:
            x3 = x3*self.tp3(x3)

        d4 = torch.cat((x3, d4), dim=1)

        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        if self.shortcut[2]:
            x2 = x2*self.tp2(x2)

        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        if self.shortcut[3]:
            x1 = x1*self.tp1(x1)

        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        return d1

if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 3, 256, 256)
    image = torch.rand(*image_size)

    # Model
    model = Model(img_ch=3, output_ch=1, shortcut=[1, 1, 1, 1], mac_pattern=0, mic_pattern=0)
    out = model(image)
    print(out.size())

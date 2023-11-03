import math
from torch import nn
import torch.nn.functional as F
import torch
import cv2
import numpy as np
from scipy import ndimage
import torchvision


def high_pass_filters(x):
    gauss = x.contiguous()
    filter = torchvision.transforms.GaussianBlur(15, sigma=5)
    gauss = filter(gauss)
    h_pass = x - gauss
    return h_pass


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.ReLU()
        )


class LSR(nn.Module):
    def __init__(self, num_channels=1, d=45, s=18, m=4):  # s原来为12
        super(LSR, self).__init__()
        self.first_part = nn.Conv2d(num_channels, d, kernel_size=3, padding=3 // 2, bias=False)
        self.prelu1 = nn.PReLU(d)
        self.mid_part = [DSLayer(d, s)]  # 56*56*1*1+3*3*56+12*1*1=3652
        for i in range(m):
            self.mid_part.extend([DSLayer1(s, s, i + 1)])  # 12*12*1*1+3*3*12+12*1*1=264
            self.mid_part.extend([CALayer(s, i + 1)])  # 12*12*1*1=144
        self.mid_part.extend([nn.Conv2d(s, 4, kernel_size=1, bias=False)])
        self.prelu2 = nn.PReLU(4)
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.first_part(x)
        x = self.prelu1(x)
        x = self.mid_part[0](x)

        for i in range(1, 9, 1):
            x = self.mid_part[i](x)

        x = self.mid_part[9](x)
        x = self.prelu2(x)
        x = self.last_part(x)
        return x


class CALayer(nn.Module):  # version 2
    def __init__(self, channel, i):
        super(CALayer, self).__init__()
        self.k = i
        self.g = nn.Sequential(nn.Conv2d(channel, channel, 1, padding=0, bias=False))
        self.v = nn.Sequential(nn.Conv2d(channel, channel, 1, padding=0, bias=False))
        self.z = nn.Sequential(nn.Conv2d(channel, channel, 1, padding=0, bias=False))

    def forward(self, x):
        y1 = self.v(x)
        y1 = F.softmax(y1, dim=1)
        y2 = self.g(x)
        out = y1 * y2
        out = self.z(out)
        return x + out


class DSLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DSLayer, self).__init__()

        layers = [ConvBNReLU(in_channel, 36, kernel_size=1)]
        layers.extend([ConvBNReLU(36, 36, stride=1, groups=36)])
        layers.extend([nn.Conv2d(36, out_channel, kernel_size=1, bias=False)])

        self.conv = nn.Sequential(*layers)
        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))

    def forward(self, x):
        x = self.conv[0](x)
        x = self.conv[1](x)
        x = self.conv[2](x)
        return x


class DSLayer1(nn.Module):
    def __init__(self, in_channel, out_channel, i):
        super(DSLayer1, self).__init__()
        self.out_channel = out_channel
        layers = [ConvBNReLU(in_channel, in_channel, stride=1, groups=in_channel)]
        layers.extend([nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)])
        self.conv = nn.Sequential(*layers)

        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))

    def forward(self, x):
        res = x
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x + res


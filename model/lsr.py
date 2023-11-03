import math
from torch import nn
import torch.nn.functional as F
import torch
import cv2
import numpy as np
from scipy import ndimage
import torchvision


class ConvReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.ReLU()
        )


class LSR(nn.Module):
    def __init__(self, num_channels=1, d=45, s=18):
        super(LSR, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=3, padding=3 // 2, bias=False),
            nn.PReLU(d)
        )
        self.LDS_A = LDS_A(d, s)
        
        self.LDSCA1 = nn.Sequential(
            LDS_B(s, s),
            CALayer(s)
        )
        self.LDSCA2 = nn.Sequential(
            LDS_B(s, s),
            CALayer(s)
        )
        self.LDSCA3 = nn.Sequential(
            LDS_B(s, s),
            CALayer(s)
        )
        self.LDSCA4 = nn.Sequential(
            LDS_B(s, s),
            CALayer(s)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(s, 4, kernel_size=1, bias=False),
            nn.PReLU(4)
        )
        self.upsample = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.LDS_A(x)

        x = self.LDSCA1(x)
        x = self.LDSCA2(x)
        x = self.LDSCA3(x)
        x = self.LDSCA4(x)

        x = self.conv2(x)
        x = self.upsample(x)
        return x


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
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


class LDS_A(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LDS_A, self).__init__()

        layers = [ConvReLU(in_channel, 36, kernel_size=1)]
        layers.extend([ConvReLU(36, 36, stride=1, groups=36)])
        layers.extend([nn.Conv2d(36, out_channel, kernel_size=1, bias=False)])

        self.layers = nn.Sequential(*layers)
        for m in self.layers:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))

    def forward(self, x):
        x = self.layers(x)
        return x


class LDS_B(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LDS_B, self).__init__()
        self.out_channel = out_channel
        layers = [ConvReLU(in_channel, in_channel, stride=1, groups=in_channel)]
        layers.extend([nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)])
        self.layers = nn.Sequential(*layers)

        for m in self.layers:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))

    def forward(self, x):
        res = x
        x = self.layers(X)
        return x + res


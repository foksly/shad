import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from collections import OrderedDict


class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.LeakyReLU(),
                 dropout_prob=0.2, bn=False):
        super().__init__()

        self.fc = nn.Linear(in_features, out_features)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_prob)
        if bn:
            self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.fc(x)
        if hasattr(self, 'batch_norm'):
            x = self.bn(x)
        x = self.activation(self.dropout(x))
        return x


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=nn.LeakyReLU(),
                 stride=1, bias=False, upsample=False, padding=None, bn=True):
        super().__init__()
        self.upsample = upsample
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding or (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=self.padding,
            bias=bias
        )

        self.bn = bn
        if self.bn:
            self.norm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=False)

        x = self.conv(x)
        x = self.activation(self.norm(x) if self.bn else x)

        return x


class Conv2dDoubleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=nn.LeakyReLU(),
                 stride=1, bias=False, upsample=False, padding=None, bn=True):
        super().__init__()

        self.conv_block1 = Conv2dBlock(in_channels, out_channels,
                                       kernel_size, activation,
                                       stride=1, bias=bias, upsample=upsample,
                                       padding=padding, bn=False)

        self.conv_block2 = Conv2dBlock(out_channels, out_channels,
                                       kernel_size, activation,
                                       stride=stride, bias=bias, upsample=False,
                                       padding=padding, bn=bn)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x

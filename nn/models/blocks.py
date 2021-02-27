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
    def __init__(self, in_features, out_features, kernel_size=3, activation=nn.LeakyReLU(),
                 stride=1, bias=False, upsample=False, norm_first=True, padding=None):
        super().__init__()
        self.upsample = upsample
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding or (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_features,
            out_features,
            kernel_size,
            stride=stride,
            padding=self.padding,
            bias=bias
        )
        self.norm = nn.BatchNorm2d(out_features)
        self.activation = activation
        self.norm_first = norm_first

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=False)

        x = self.conv(x)
        x = self.activation(self.norm(x)) if self.norm_first else self.norm(self.activation(x))

        return x


class Conv2dDenoisingBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, activation=nn.LeakyReLU(),
                 stride=1, bias=False, upsample=False, norm_first=True, dropout_prob=0.3):
        super().__init__()
        self.upsample = upsample

        self.conv = nn.Conv2d(
            in_features,
            out_features,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=bias
        )
        self.norm = nn.BatchNorm2d(out_features)
        self.dropout = nn.Dropout(0.3)
        self.activation = activation
        self.norm_first = norm_first

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        x = x + torch.randn_like(x) * 0.05
        x = self.dropout(x)

        x = self.conv(x)
        x = self.activation(self.norm(x)) if self.norm_first else self.norm(self.activation(x))

        return x

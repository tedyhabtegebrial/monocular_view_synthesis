"""
copyright:
author: "Tewodros Amberbir Habtegebrial"
license: "MIT"
email: "tedyhabtegebrial@gmail.com"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """ docstring for ResBlock."""

    def __init__(self, in_ch, k):
        super(ResBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_ch, in_ch//2, kernel_size=k, stride=1, padding=k // 2)
        self.conv_2 = nn.Conv2d(in_ch // 2, in_ch, kernel_size=k, stride=1, padding=k // 2)
    def forward(self, input_):
        out_1 = F.relu(self.conv_1(input_))
        out_1 = F.relu(self.conv_2(out_1))
        out = out_1 + input_
        return out

class ConvBlock(nn.Module):
    def __init__(self, inp_chans, out_chans, k_size, down_sample=True, use_no_relu=False):
        super(ConvBlock, self).__init__()
        stride_0 = 2 if down_sample else 1
        self.conv_0 = nn.Conv2d(inp_chans, out_chans, kernel_size=k_size, stride=stride_0, padding=k_size // 2)
        self.conv_1 = nn.Conv2d(out_chans, out_chans, kernel_size=k_size, stride=1, padding=k_size // 2)
        nn.init.xavier_normal_(self.conv_0.weight.data, gain=1.0)
        nn.init.xavier_normal_(self.conv_1.weight.data, gain=1.0)
        self.non_linearity_0 = F.relu
        self.non_linearity_1 = F.relu
        if use_no_relu:
            self.non_linearity_1 = nn.Sequential()
        else:
            self.non_linearity_1 = F.relu

    def forward(self, x):
        x1 = self.non_linearity_0(self.conv_0(x))
        x2 = self.non_linearity_1(self.conv_1(x1))
        return x2

class DeconvBlock(nn.Module):
    def __init__(self, inp_chans, out_chans):
        super(DeconvBlock, self).__init__()
        self.conv_0 = nn.Conv2d(inp_chans, out_chans, kernel_size=3, stride=1, padding=1)
        self.non_linearity = nn.ReLU(True)

    def forward(self, inputs):
        x = torch.cat(inputs, dim=1) if isinstance(inputs, list) else inputs
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv_0(x))
        return x

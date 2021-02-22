#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 15:19:27 2021

@author: nuvilabs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
        """It only support square kernels and stride=1, dilation=1, groups=1."""
        super(Conv2dSame, self).__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.net(x)
class SudokuCNN(nn.Module):
    def __init__(self):
        super(SudokuCNN, self).__init__()
        self.conv_layers = nn.Sequential(Conv2dSame(1,512,3), #1
                                         Conv2dSame(512,512,3),#2
                                         Conv2dSame(512,512,3),#3
                                         Conv2dSame(512,512,3),#4
                                         Conv2dSame(512,512,3),#5
                                         Conv2dSame(512,512,3),#6
                                         Conv2dSame(512,512,3),#7
                                         Conv2dSame(512,512,3),#8
                                         Conv2dSame(512,512,3),#9
                                         Conv2dSame(512,512,3),#10
                                         Conv2dSame(512,512,3),#11
                                         Conv2dSame(512,512,3),#12
                                         Conv2dSame(512,512,3),#13
                                         Conv2dSame(512,512,3),#14
                                         Conv2dSame(512,512,3))#15
        self.last_conv = nn.Conv2d(512, 9, 1)
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.last_conv(x)
        return x
        

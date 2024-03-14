#!~/opt/anaconda3/bin/python
# -*- coding: utf-8 -*-
# @Time : 2022/10/5 23:15
# @Author: Mark Zeng
# @Software: PyCharm


import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict


class ECABlock(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avgpool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

        # y = self.avgpool(x)  # bs,c,1,1
        # y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        # y = self.conv(y)  # bs,1,c
        # y = self.sigmoid(y)  # bs,1,c
        # y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        # return x * y.expand_as(x)


if __name__ == '__main__':
    input = torch.randn(50, 256, 8, 8)
    eca = ECABlock(kernel_size=3)
    output = eca(input)
    print(output.shape)

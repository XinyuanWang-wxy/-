# -*- coding: utf-8 -*-
# @Time : 2022/10/4 17:10
# @Author: Yukai Zeng


from torch import nn
from SeNet import SeBlock
from ECANet import ECABlock
from CBAM import CBAMBlock
from BAM import BAMBlock
from SANet import ShuffleAttention


def AM():
    SeNet = SeBlock(channel=256, reduction=8),
    CBAM = CBAMBlock(channel=256, reduction=16, kernel_size=7),
    ECANet = ECABlock(kernel_size=7),
    BAM = BAMBlock(channel=256, reduction=16, dia_val=1),
    SANet = ShuffleAttention(channel=256, G=8),


class Net_1(nn.Module):
    def __init__(self):
        super(Net_1,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1), nn.ReLU(),)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1), nn.ReLU(),)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1), nn.ReLU(),)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1), nn.ReLU(),)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(256 * 8 * 8, 256), nn.ReLU(),)
        self.dropout1 = nn.Dropout(p=0.50)
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),)
        self.dropout2 = nn.Dropout(p=0.50)
        self.fc3 = nn.Sequential(nn.Linear(128, 2), nn.Sigmoid(),)

        self.all_layer = nn.Sequential(self.conv1, self.maxpool1,
                                       self.conv2, self.maxpool2,
                                       self.conv3, self.maxpool3,
                                       self.conv4, self.maxpool4,
                                       self.flatten, self.fc1, self.dropout1, self.fc2, self.dropout2, self.fc3)

    def forward(self, x):

        # output = [] # 记录各层网格特征
        # for layer in self.all_layer:
        #     x = layer(x)
        #     output.append(x)
        # return output

        for layer in self.all_layer:
            x = layer(x)
        return x


# class Net_1(nn.Module): # 对应CAM结构
#     def __init__(self):
#         super(Net_1,self).__init__()
#         self.conv1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1), nn.ReLU(),)
#         self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1), nn.ReLU(),)
#         self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1), nn.ReLU(),)
#         self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1), nn.ReLU(),)
#         self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(256*1*1, 5)
#         self.sigmoid = nn.Sigmoid()

#         self.all_layer = nn.Sequential(self.conv1, self.maxpool1,
#                                        self.conv2, self.maxpool2,
#                                        self.conv3, self.maxpool3,
#                                        self.conv4, self.maxpool4, SeBlock(channel=256, reduction=8),
#                                        self.avg_pool, self.flatten, self.fc, self.sigmoid)

#     def forward(self, x):
#         for layer in self.all_layer:
#             x = layer(x)
#         return x
'''
Author: XinyuanWang
Email: zero1730816185@163.com
Date: 2024-03-15 23:19:34
LastEditors: XinyuanWang zero1730816185@163.com
LastEditTime: 2024-03-16 23:39:08
'''

import torch
import torch.nn as nn

class Cnn1d(nn.Module):
    def __init__(self):
        super(Cnn1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(2498, 2)
        self.dropout1=nn.Dropout(0.25)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.pool2(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = torch.tanh(x)
        x = self.pool3(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        return x
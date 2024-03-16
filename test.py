'''
Author: XinyuanWang
Email: zero1730816185@163.com
Date: 2024-03-16 01:38:11
LastEditors: XinyuanWang zero1730816185@163.com
LastEditTime: 2024-03-16 23:33:36
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
import pandas as pd
from Net import Cnn1d
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
file_path = "E:\\tasks\SCI噪声探测多分类\data\data_dalian_checked\wavs_data\\20210731030000-2105000025-nb.wav"
signal, sr = librosa.load(file_path, sr=None)
spectrum = np.fft.fft(signal,2000*5)
spectrum = np.abs(spectrum)
test=torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0)
print(test)
module = Cnn1d(5)
module.load_state_dict(torch.load('model.pth'))
outputs = module(test)
_, predicted = torch.max(outputs, 1)
print(predicted)
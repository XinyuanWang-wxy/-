'''
Author: XinyuanWang
Email: zero1730816185@163.com
Date: 2024-03-15 16:57:47
LastEditors: XinyuanWang zero1730816185@163.com
LastEditTime: 2024-03-17 02:49:38
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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 定义数据集类
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # 加载音频文件
        signal, sr = librosa.load(file_path, sr=None)
        spectrum = np.fft.fft(signal,20000)#提取0~4000HZ的频率幅值数据
        spectrum = np.abs(spectrum)
        return torch.tensor(spectrum, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 构建1D-CNN模型




# 加载.wav文件和标签
def load_data(data_dir):
    file_paths = []
    labels_dict = {}
    labels_excel = "../data/data_dalian_checked/wavs_label.xlsx"
    df=pd.read_excel(labels_excel)
    for index, row in df.iterrows():
        file_name = row['wav_name']
        label = row['label']
        file_path = os.path.join(data_dir,file_name)
        labels_dict[file_path] = label
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        file_paths.append(file_path)
    return file_paths, labels_dict

# 数据集目录
data_dir = '../data/data_dalian_checked/wavs_data/'

# 加载数据
file_paths, labels_dict = load_data(data_dir)
labels=[]
for file in file_paths:
    labels.append(labels_dict[file])
# 划分训练集和测试集
train_file_paths, test_file_paths, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

# 创建数据集和数据加载器
train_dataset = AudioDataset(train_file_paths, train_labels)
test_dataset = AudioDataset(test_file_paths, test_labels)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
device = torch.device(DEVICE)
# 初始化模型、损失函数和优化器
module = Cnn1d().to(device)
try:
    module.load_state_dict(torch.load(f"model_{DEVICE}.pth"))
except:
    print("未加载到模型文件")
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = optim.Adam(module.parameters(), lr=0.001)

# 训练模型
num_epochs = 1
epoch_loss_series = []
for epoch in range(num_epochs):
    module.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = module(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    epoch_loss_series.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}')
torch.save(module.state_dict(), f"model_{DEVICE}.pth")

plt.plot(range(1,num_epochs+1),epoch_loss_series)
plt.savefig("train_loss.png")
# 在测试集上评估模型
module.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = module(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')


# @Time : 2024/03/07
# @Author: Yukai Zeng


import os
import h5py
import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split


def save_dataset_to_h5py(x_data, y_data, name, func_name):
    """
    保存数据集
    :param x_data: 处理后data
    :param y_data: 标签
    :param name: train/test
    :param func_name: 信号处理方式名
    """

    shuffle = np.random.permutation(x_data.shape[0]) # 打乱顺利
    x_data = x_data[shuffle]
    y_data = y_data[shuffle]

    os.makedirs('./dataset/', exist_ok=True)
    f = h5py.File(f'./dataset/signs_grey_{func_name}_{name}.h5', 'w')  # 写入文件
    f[f'x_{name}'] = x_data
    f[f'y_{name}'] = y_data
    f.close()  # 关闭文件


def cwt_image_data(data, fs, wavelet_name, jpg_path):
    if not os.path.exists(jpg_path):
        t = np.arange(0, len(data)) / fs
        totalscale = 128
        fc = pywt.central_frequency(wavelet_name)  # 中心频率
        cparam = 2 * fc * totalscale
        scales = cparam / np.arange(totalscale, 0, -1) # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
        cwtmatr, frequencies = pywt.cwt(data, scales, wavelet_name, 1.0 / fs)  # 连续小波变换

        # CWT生成图像
        plt.figure()
        plt.contourf(t, frequencies, np.abs(cwtmatr))
        # plt.ylim([200, 2000])  
        plt.axis('off')
        # plt.tight_layout()
        plt.savefig(jpg_path, bbox_inches='tight', pad_inches = 0)
        plt.close()

    # 读取数据
    # https://blog.csdn.net/enter89/article/details/90262569
    image = Image.open(jpg_path).convert('RGB')
    image = image.resize((128, 128)) # 尺寸
    # https://www.cnblogs.com/haifwu/p/12855563.html
    image = transforms.Compose([transforms.Grayscale(1)])(image) # 转成灰度
    data_image = np.array(image)  # 转化成数组
    return data_image


def extract_dat_file(file_path):
    # Read the binary data from the file
    with open(file_path, 'rb') as f:
        A = np.fromfile(f, dtype=np.float32)
    # Reshape the array
    A = A.reshape(-1, 1)

    # Extract header, data, sampling frequency, and time
    Header = A[:64].reshape(-1)
    Data = A[64:].reshape(-1)
    fs = A[10][0]  # Assuming 0-based indexing in Python
    time = A[17][0]  # Assuming 0-based indexing in Python
    print(fs,time,file_path)
    pulses = int(fs * time)
    data_length = len(Data) // pulses
    B = np.zeros((pulses, data_length))
    for i in range(pulses):
        B[i, :] = Data[data_length * i:data_length * (i + 1)]

    t = np.arange(1, pulses + 1) / fs
    return int(fs), B


def generate_dataset(dat_path_list, leak_channels, normal_channels, wavelet_name='morl'): # 生成数据集并保存
    data_x, data_y = [], []
    os.makedirs(f'./image/{wavelet_name}_signs/', exist_ok=True)
    for dat_path in dat_path_list: # 遍历实验组dat文件
        fs, B = extract_dat_file(dat_path) # 读取数据 data_channels
        hour, minute, second = dat_path.split('/')[-1].split('-')[3:6]
        file_name = f'{hour}-{minute}-{second}'
        seconds = int(B.shape[0]/fs)

        for channel in leak_channels:
            for second in range(seconds):
                data = B[:, channel][fs*second:fs*(second+1)]
                jpg_path = f'./image/{wavelet_name}_signs/{file_name}_channel{channel}_{second}.jpg'
                data_x.append(cwt_image_data(data, fs, wavelet_name, jpg_path))
                data_y.append(1)
        
        for channel in normal_channels:
            for second in range(seconds):
                data = B[:, channel][fs*second:fs*(second+1)]
                jpg_path = f'./image/{wavelet_name}_signs/{file_name}_channel{channel}_{second}.jpg'
                data_x.append(cwt_image_data(data, fs, wavelet_name, jpg_path))
                data_y.append(0)

    x_train, x_test, y_train, y_test = train_test_split(np.array(data_x), np.array(data_y), test_size=0.2, random_state=2023, stratify=data_y) # 划分训练集和测试集，每种类别的样本比例相同

    print(f"训练集：{x_train.shape[0]}个数据")
    save_dataset_to_h5py(x_train, y_train, 'train', wavelet_name)
    print(f"测试集：{x_test.shape[0]}个数据")
    save_dataset_to_h5py(x_test, y_test, 'test', wavelet_name)


def main():
    # 泄漏信道: 33、34
    # 正常信道: 28、39
    leak_channels = [33, 34]
    normal_channels = [28, 39]
    time_check = []  # 时间检查

    df_case = pd.read_excel('./case_info.xlsx') # 实验组信息

    for time_start, lasting_seconds, time_lag_seconds in zip(df_case['时间起'].astype(str).values, df_case['时长'].values, df_case['设备时差'].values):
        time_check_start = datetime.strptime(time_start, '%Y-%m-%d-%H-%M-%S') + timedelta(seconds=int(time_lag_seconds))
        time_check_end = time_check_start + timedelta(seconds=int(lasting_seconds))
        time_check.append((time_check_start, time_check_end))

    # 从数据中挑出数据组数据
    dat_path_list = [] # dat文件路径
    for file_name in sorted(os.listdir(f'./1.4数据/')): # 数据文件
        tem=file_name.split('-out')[0]
        data_time = datetime.strptime(tem, '%Y-%m-%d-%H-%M-%S')
        for time_check_start, time_check_end in time_check:
            if time_check_start <= data_time < time_check_end: # 满足时间条件，对应实验数据
                dat_path_list.append(f'./1.4数据/{file_name}')
                break

    print(f"***** {len(dat_path_list)}个dat文件 *****")
    generate_dataset(dat_path_list, leak_channels, normal_channels)


if __name__ == '__main__':
    main()

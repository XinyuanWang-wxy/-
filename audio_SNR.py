'''
Author: XinyuanWang
Email: zero1730816185@163.com
Date: 2024-03-15 15:17:27
LastEditors: XinyuanWang zero1730816185@163.com
LastEditTime: 2024-03-15 16:33:08
'''

import os
import librosa
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

def calculate_snr(signal, noise):
    # 计算信号的功率
    signal_power = np.sum(np.square(signal))

    # 计算噪声的功率
    noise_power = np.sum(np.square(noise))

    # 计算信噪比（SNR）
    snr = 10 * np.log10(signal_power / noise_power)

    return snr

# 声音文件所在目录
audio_directory = '../data/data_dalian_checked/wavs_data/'
snrs=[]
# 获取目录下所有声音文件
audio_files = [f for f in os.listdir(audio_directory) if f.endswith('.wav')]
# 将字符串写入文本文件
filename = "snr.txt"
if os.path.exists(filename):
# 如果文件存在，则删除文件
    os.remove(filename)
# 循环处理每个声音文件
for audio_file in audio_files:
    # 读取声音文件
    audio_path = os.path.join(audio_directory, audio_file)
    signal, sr = librosa.load(audio_path, sr=None)

    # 使用滤波器分离信号和噪声
    # 这里使用简单的低通滤波器作为示例，您可以根据实际情况选择更适合的滤波器
    b, a = scipy.signal.butter(4, 0.1, 'low')  # 设计4阶低通滤波器，截止频率为0.1倍采样率
    filtered_signal = scipy.signal.filtfilt(b, a, signal)
    filtered_noise = signal - filtered_signal

    # 计算信噪比
    snr = calculate_snr(filtered_signal, filtered_noise)
    snrs.append(snr)
    text = "File: {}, Signal to Noise Ratio (SNR): {:.2f} dB\n".format(audio_file, snr)


    with open(filename, "a") as file:
        file.write(text)
# 生成直方图
plt.hist(snrs, bins=5, color='skyblue', edgecolor='black')

# 添加标题和标签
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# 保存直方图
plt.savefig('snr_hist.png')
# 显示图形
plt.show()
print("completed")

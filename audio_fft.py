'''
Author: XinyuanWang
Email: zero1730816185@163.com
Date: 2024-03-15 20:43:36
LastEditors: XinyuanWang zero1730816185@163.com
LastEditTime: 2024-03-17 00:03:31
'''

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
#读取多条声音文件并进行傅里叶变换
def process_audio_files(audio_dir, output_dir):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取目录下所有声音文件
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

    # 循环处理每个声音文件
    for audio_file in audio_files:
        # 读取声音文件
        audio_path = os.path.join(audio_dir, audio_file)
        signal, sr = librosa.load(audio_path, sr=None)

        # 进行傅里叶变换
        print(len(signal))
        fft_result = np.fft.fft(signal)
        print(len(fft_result))
        frequencies = np.fft.fftfreq(len(signal), d=1/sr)
        print(len(frequencies))

        magnitude = np.abs(fft_result)
        # 绘制傅里叶变换后的图像
        plt.figure(figsize=(8, 4))
        plt.plot(frequencies, magnitude)
        plt.title('FFT Result')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(0,2000)
        print(np.max(magnitude[1:]))
        plt.ylim(0,max(magnitude[1:]))
        plt.grid(True)

        # 保存图像
        output_file = os.path.join(output_dir, os.path.splitext(audio_file)[0] + '_fft.png')
        plt.savefig(output_file)
        plt.close()
# 指定声音文件目录和输出目录
audio_dir = '../data/data_dalian_checked/wavs_data/'
output_dir = '../img/fft'
# 处理声音文件并保存傅里叶变换图像
process_audio_files(audio_dir, output_dir)
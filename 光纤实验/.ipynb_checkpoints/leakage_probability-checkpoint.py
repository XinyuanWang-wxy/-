# Time: 2024-03-08
# Author: Yukai Zeng


import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Net import Net_1
from datetime import datetime, timedelta
from dataset_generate import extract_dat_file, cwt_image_data


def get_lab_data():
    time_check = [] # 时间检查
    df_case = pd.read_excel('./case_info.xlsx') # 实验组信息
    for time_start, lasting_seconds, time_lag_seconds in zip(df_case['时间起'].astype(str).values, df_case['时长'].values, df_case['设备时差'].values):
        time_check_start = datetime.strptime(time_start, '%Y-%m-%d-%H-%M-%S') + timedelta(seconds=int(time_lag_seconds))
        time_check_end = time_check_start + timedelta(seconds=int(lasting_seconds))
        time_check.append((time_check_start, time_check_end))

    # 从数据中挑出数据组数据
    dat_path_list = [] # dat文件路径
    for file_name in sorted(os.listdir(f'./1.4数据')): # 数据文件
        data_time = datetime.strptime(file_name.split('-out')[0], '%Y-%m-%d-%H-%M-%S')
        for time_check_start, time_check_end in time_check:
            if time_check_start <= data_time < time_check_end: # 满足时间条件，对应实验数据
                dat_path_list.append(f'./1.4数据/{file_name}')
                break
    return dat_path_list


def load_model(DEVICE, func_name='morl', BATCH_SIZE=256, EPOCH=250, LR=0.00005):
    model_path = f'./model/{func_name}/grey-Net_B{BATCH_SIZE}-E{EPOCH}-L{LR}.pth'

    # 读取模型
    try:
        model_state_dict = torch.load(model_path) # map_location='cpu'
    except:
        raise Exception(f"{model_path} not found!")
    
    model = Net_1().to(DEVICE) # 模型类实例化
    model.load_state_dict(model_state_dict) # 加载模型参数
    return model


def get_leakage_probability(model, x_data, DEVICE):
        x = torch.FloatTensor(x_data.reshape((1, x_data.shape[0], x_data.shape[1]))) # 转换为张量
        model_result = model(x.unsqueeze(0).to(DEVICE))
        leakage_probability = model_result.cpu().data.numpy().flatten()[1]
        return leakage_probability


def get_probabilities(dat_path_list, model, DEVICE, channel_max=57, wavelet_name='morl'): # [channel, time, leakage_probability]
    result = []
    channel_list = range(1, channel_max+1) # 信道列表
    os.makedirs(f'./image/{wavelet_name}_signs/', exist_ok=True)
    for dat_path in dat_path_list: # 遍历实验组dat文件
        fs, B = extract_dat_file(dat_path) # 读取数据 data_channels
        hour, minute, second = dat_path.split('/')[-1].split('-')[3:6]
        file_name = f'{hour}-{minute}-{second}'
        seconds = int(B.shape[0]/fs)

        for channel in channel_list:
            result_channel = []
            for second in range(seconds):
                data = B[:, channel][fs*second:fs*(second+1)]
                jpg_path = f'./image/{wavelet_name}_signs/{file_name}_channel{channel}_{second}.jpg'
                data_image = cwt_image_data(data, fs, wavelet_name, jpg_path)
                result_channel.append(get_leakage_probability(model, data_image, DEVICE)) # 每秒泄漏概率
            result.append([channel, file_name, np.array(result_channel).mean()])

        print(f"{file_name} done!")
    result = sorted(result, key=lambda x: -x[-1]) # 按泄漏概率逆序排序
    return result


def draw_map(df): # 作泄漏风险图
    # columns=['channel', 'time', 'leakage_probability']

    channel_array = df['channel'].astype(int) # x
    time_dict = {time: i+1 for i, time in enumerate(sorted(df['time'].unique(), reverse=True))}
    time_array = df['time'].map(time_dict) # y
    value_array = df['leakage_probability'].astype(float) # z
    min_value = value_array.min()
    max_value = value_array.max()

    # 散点图
    plt.figure(figsize=(16, 16))
    plt.scatter(
        channel_array, # x-axis
        time_array, # y-axis
        c=value_array, # value
        cmap='viridis_r', # 颜色映射 'cividis', 'viridis'
        s=100, # 散点大小
        alpha=0.9,  # 透明度
        edgecolors='w',  # 散点的边缘颜色
        linewidth=0.2,  # 散点的边缘宽度
        vmin=min_value,  # Set minimum value for color scale
        vmax=max_value  # Set maximum value for color scale
    )
    
    # 添加网格
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    # 添加坐标轴标签和标题
    plt.xlabel('Channel')
    plt.ylabel('Time')
    plt.title(f'Pipeling Leakage Probability', loc='center')
    # 添加颜色条和标签
    cbar = plt.colorbar()
    cbar.set_label('leakage probability', rotation='vertical')
    # 添加y轴刻度显示文字
    y_ticks = list(time_dict.values())
    y_tick_labels = list(time_dict.keys())
    plt.yticks(y_ticks, y_tick_labels) 
    # 设置y轴刻度标签的字体大小
    plt.tick_params(axis='y', labelsize=6)
    # 保存图形
    plt.savefig(f'./result/probability_data.png')
    plt.close()


def main():
    dat_path_list = get_lab_data() # 实验组数据

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # GPU
    model = load_model(DEVICE) # 加载模型
    
    try:
        df = pd.read_excel(f'./result/probability_data.xlsx')
        if df.shape[0] == 0:
            raise Exception
    except:
        probabilites_result = get_probabilities(dat_path_list, model, DEVICE)
        df = pd.DataFrame(probabilites_result, columns=['channel', 'time', 'leakage_probability'])
        os.makedirs(f'./result/', exist_ok=True)
        df.to_excel(f'./result/probability_data.xlsx', index=False)
    
    draw_map(df) # 作泄漏风险图


if __name__ == '__main__':
    main()
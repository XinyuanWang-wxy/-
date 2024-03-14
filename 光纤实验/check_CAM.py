# -*- coding: utf-8 -*-
# @Time: 2023.03.11
# @Author: Yukai Zeng


import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from PIL import Image
from d2l import torch as d2l


def middle_layer_value(model, layer, x):
    for i in range(layer):
        x = model.all_layer[i](x)
    return x

    
def check_CAM(test_loader, model, model_name, func_name, acc, image_size, DEVICE, layer, check_number=15):
    """
    layers pay attention to
    https://www.jianshu.com/p/41d85603ccc1
    https://blog.csdn.net/qq_26024067/article/details/118660764
    https://colab.research.google.com/github/zaidalyafeai/AttentioNN/blob/master/Attention_Maps.ipynb
    :param test_loader: testset dataloader
    :param model: 模型
    :param model_name: 名字
    :param func_name: 方式名
    :param acc: 模型精度
    :param image_size: 图片大小
    :param DEVICE: gpu
    :param layer: 模型GAP前一层的位序
    :param check_number: 3的倍数
    """
    num_loader = np.random.randint(1, test_loader.__len__())
    for _ in range(num_loader):
        X, y = next(iter(test_loader))
    count_left = X.shape[0]*num_loader

    start = np.random.randint(X.shape[0]-check_number)
    os.makedirs(f'./image/capture_{func_name}/{model_name}/', exist_ok=True)
    print(f"test_loader data: {count_left+start}~{count_left+start+check_number}")

    # 保存原图
    plt.figure(1)
    for n in range(check_number):
        axe = plt.subplot(3, int(check_number/3), n + 1)
        axe.imshow(Image.fromarray(np.uint8(X)[start+n].reshape(image_size))) 
        axe.set_title(f'label={np.array(y)[start+n]}', fontsize=8)
        axe.set_axis_off()
    plt.suptitle(f'original image(test_acc={acc:.3f})')
    plt.tight_layout()
    plt.savefig(f'./image/capture_{func_name}/{model_name}/start{count_left+start}_original.png', dpi=300)
    plt.close()

    # CAM作图
    feature_layer = middle_layer_value(model, layer, X.to(DEVICE)).cpu().data.numpy()
    y_hat = d2l.argmax(model(X.to(DEVICE)), axis=1).cpu().data.numpy()
    channel_weight = model._modules.get('fc').weight.cpu().data.numpy()[y_hat] # 选取预测类别的权重

    features = feature_layer[start:start+check_number]
    weights = channel_weight[start:start+check_number]

    heatmap_layer = [] # CAM数据
    for n, (weight_n, feature_n) in enumerate(zip(weights, features)):
        c, h, w = feature_n.shape
        cam = np.zeros((h, w))
        for weight, feature in zip(weight_n, feature_n):
            cam += weight * np.abs(feature)
        cam = cv2.resize(cam, image_size)
        cam = (1 - cam / cam.max()) 
        cam_grey = np.uint8(255 * cam) # 灰度图
        heatmap = cv2.applyColorMap(cam_grey, cv2.COLORMAP_JET)
        # image = np.uint8(X)[start+n].reshape((128, 128, 3))
        # heatmap = cv2.addWeighted(heatmap, 0.6, image, 0.4, 0)
        heatmap_layer.append(heatmap)

    # 作图
    plt.figure(2)
    for i in range(check_number):
        axe = plt.subplot(3, int(check_number/3), i + 1)
        axe.imshow(heatmap_layer[i])
        axe.set_title(f'label={np.array(y)[start+i]} predict={y_hat[start+i]}', fontsize=8)
        axe.set_axis_off()
    plt.suptitle(f'CAM results')
    plt.tight_layout()
    plt.savefig(f'./image/capture_{func_name}/{model_name}/start{count_left+start}_CAM_layer{layer}.png', dpi=300)
    plt.close()
    print('figure_CAM done')


if __name__ == '__main__':
    pass

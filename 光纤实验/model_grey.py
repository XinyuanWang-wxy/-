# -*- coding: utf-8 -*-
# @Time: 2022/9/26 13:24
# @Author: Yukai Zeng


import os
import torch
from torch import nn
from torchviz import make_dot
from torchsummary import summary
from dataset import load_dataset
from torch_structure import evaluate_accuracy_loss, net_train, net_train_early_stopping
from Net import Net_1
# from check_CAM import check_CAM
from itertools import product


def layer_shape(net, shape, DEVICE):
    X = torch.randn(shape, dtype=torch.float32).to(DEVICE) # 标准正太分布
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape: \t', X.shape)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def main(func_name, model_name, BATCH_SIZE, EPOCH, LR, layer):
    epoch_decay_start = EPOCH # LR衰减, beta1变小

    # 加载数据集
    train_loader, test_loader = load_dataset('signs_grey', BATCH_SIZE, func_name)
    # train_loader, valid_loader, test_loader = load_dataset('signs_grey', BATCH_SIZE, func_name, valid_size=0.25)
    image_size = (128, 128)

    # 创建模型保存路径
    os.makedirs(f'./model/{func_name}/', exist_ok=True)
    model_path = f'./model/{func_name}/{model_name}.pth'

    if not os.path.exists(model_path):  # 模型未训练
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # GPU
        print(f'Training on {DEVICE}')
        model = Net_1().to(DEVICE) # 模型类实例化
        model.apply(init_weights) # 初始化权重
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # 定义优化器
        loss_function = nn.CrossEntropyLoss()  # 定义损失函数，交叉熵损失函数

        # 展示各层维度变化
        # summary(model.all_layer, (1,)+image_size)
        # layer_shape(model.all_layer, (1, 1)+image_size, DEVICE)

        # Adjust learning rate and betas for Adam Optimizer
        momentum1 = 0.9
        momentum2 = 0.1
        alpha_plan = [LR] * EPOCH
        beta1_plan = [momentum1] * EPOCH
        for i in range(epoch_decay_start, EPOCH):
            alpha_plan[i] = float(EPOCH - i) / (EPOCH - epoch_decay_start) * LR
            beta1_plan[i] = momentum2

        # 训练
        net_train(model, model_name, func_name, alpha_plan, beta1_plan, train_loader, test_loader, loss_function, optimizer, EPOCH, DEVICE)
        # net_train_early_stopping(model, model_name, func_name, alpha_plan, beta1_plan, train_loader, valid_loader, test_loader, loss_function, optimizer, EPOCH, DEVICE, patience=15)
       
        # 保存模型参数
        torch.save(model.state_dict(), model_path)
        # 保存模型结构 https://www.freesion.com/article/340667237/
        # make_dot(model(torch.randn((1, 1, 128, 128)).to(DEVICE))[-1]).render(f'./data/{func_name}/models/{model_name}', view=False)
    else:
        print("load model_state_dict")
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # GPU
        model_state_dict = torch.load(model_path) # map_location='cpu'
        model = Net_1().to(DEVICE) # 模型类实例化
        model.load_state_dict(model_state_dict) # 加载模型参数
        test_acc = evaluate_accuracy_loss(model, test_loader, DEVICE) # 测试集精度
        print(f"test_acc = {test_acc:.3f}")
        # 查看各层特征
        # check_CAM(test_loader, model, model_name, func_name, test_acc, image_size, DEVICE, layer, check_number=12)


if __name__ == '__main__':

    func_name = 'morl'
    model_name = 'grey-Net'
    layer = 8
    # model_name = 'grey-SeNet'
    # layer = 9

    BATCH_SIZE = 256 # 批次大小
    EPOCH = 250 # 训练轮数
    LR = 0.00005 # 初始学习率
    model_name_new = model_name + f'_B{BATCH_SIZE}-E{EPOCH}-L{LR}'
    main(func_name, model_name_new, BATCH_SIZE, EPOCH, LR, layer)

    # 参数试验
    # func_name = 'morl'
    # model_name = 'grey-Net'
    # layer = 8
    # # model_name = 'grey-SeNet'
    # # layer = 9

    # BATCH_SIZE_list = [64, 128, 256]
    # EPOCH_list = [150]
    # LR_list = [0.000008, 0.00001, 0.00005, 0.00008, 0.0001, 0.0005, 0.0008, 0.001, 0.005]

    # for BATCH_SIZE, EPOCH, LR in product(BATCH_SIZE_list, EPOCH_list, LR_list):
    #     model_name_new = model_name + f'_B{BATCH_SIZE}-E{EPOCH}-L{LR}'
    #     main(func_name, model_name_new, BATCH_SIZE, EPOCH, LR, layer)
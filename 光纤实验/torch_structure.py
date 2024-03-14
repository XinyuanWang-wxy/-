# -*- coding: utf-8 -*-
# @Time : 2022/10/19 13:35
# @Author: Yukai Zeng


import torch
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l
from pytorchtools import EarlyStopping


def evaluate_accuracy_loss(net, data_iter, device=None, loss_function=None, loss=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # （损失之和,） 正确预测的数量，总数量
    metric = d2l.Accumulator(3) if loss else d2l.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            # BERT微调所需的（之后将介绍）
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        if loss:
            metric.add(loss_function(y_hat, y).item() * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
        else:
            metric.add(d2l.accuracy(y_hat, y), X.shape[0])
    return (metric[0] / metric[2], metric[1] / metric[2]) if loss else metric[0] / metric[1]


def adjust_learning_rate(optimizer, alpha_plan, beta1_plan, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1


def net_train(model, model_name, func_name, alpha_plan, beta1_plan, train_loader, test_loader, loss_function, optimizer, EPOCH, DEVICE):
    timer = d2l.Timer()

    lr_list = [] # 记录学习率变化
    train_loss_list = [] # 记录训练损失变化
    train_acc_list = [] # 记录训练准确率变化
    test_acc_list = [] # 记录测试准确率变化

    for epoch in range(EPOCH):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        model.train()
        adjust_learning_rate(optimizer, alpha_plan, beta1_plan, epoch)
        lr_list.append((optimizer.state_dict()['param_groups'][0]['lr']))
        for (X, y) in train_loader: # 一次迭代一个batch
            timer.start()
            optimizer.zero_grad() # 清空上一次梯度
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_hat = model(X)
            loss = loss_function(y_hat, y) # 计算误差
            loss.backward() # 误差反向传递
            optimizer.step() # 优化器参数更新
            metric.add(loss.item() * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()

        train_loss = metric[0] / metric[2] # 训练损失
        train_acc = metric[1] / metric[2] # 训练集准确率
        test_acc = evaluate_accuracy_loss(model, test_loader, DEVICE) # 测试集准确率
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f'epoch[{epoch+1}/{EPOCH}] train_loss: {train_loss:.3f} train_acc: {train_acc:.3f} test_acc: {test_acc:.3f}')
    
    # 训练过程可视化
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_loss_list)), train_loss_list, color='b', linestyle='-', label='train loss')
    plt.plot(range(len(train_acc_list)), train_acc_list, color='m', linestyle='--', label='train acc')
    plt.plot(range(len(test_acc_list)), test_acc_list, color='g', linestyle='-.', label='test acc')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss & Accuracy')
    plt.title('Loss & Accuracy Change Over Epochs')
    plt.savefig(f'./model/{func_name}/loss-acc_{test_acc:.3f}_{model_name}.png', dpi=500)
    plt.clf() # 清除图像
    print(f'{metric[2] * EPOCH / timer.sum():.1f} examples/sec training on {str(DEVICE)}')
    
    # 学习率变化曲线
    # plt.plot(range(len(lr_list)), lr_list, color='b', linestyle='-', label='Learning Rate')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend()
    # plt.xlabel('Epoch')
    # plt.ylabel('Learning Rate')
    # plt.title('Learning Rate Change Over Epochs')
    # plt.savefig(f'./model/{func_name}/lr-change_{test_acc:.3f}_{model_name}.png', dpi=500)
    # plt.close()


def net_train_early_stopping(model, model_name, func_name, alpha_plan, beta1_plan, train_loader, valid_loader, test_loader, loss_function, optimizer, EPOCH, DEVICE, patience=10):
    animator = d2l.Animator(xlabel='epoch', xlim=[1, EPOCH],
                            legend=['train loss', 'valid loss', 'train acc', 'valid acc', 'test acc'],
                            fmts=('-', 'b-', 'm--', 'g-.', 'r:'), figsize=(8, 5))
    timer, num_batches = d2l.Timer(), len(train_loader)

    # initialize the early_stopping object
    # https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    lr_list = [] # 记录学习率变化
    for epoch in range(EPOCH):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        model.train()
        adjust_learning_rate(optimizer, alpha_plan, beta1_plan, epoch)
        lr_list.append((optimizer.state_dict()['param_groups'][0]['lr']))
        for step, (X, y) in enumerate(train_loader):
            timer.start()
            optimizer.zero_grad() # 清空上一次梯度
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_hat = model(X)
            loss = loss_function(y_hat, y) # 计算误差
            loss.backward() # 误差反向传递
            optimizer.step() # 优化器参数更新
            metric.add(loss.item() * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (step + 1) % (num_batches // 5) == 0 or step == num_batches - 1:
                animator.add(epoch + (step + 1) / num_batches, (train_loss, None, train_acc, None, None))
        # 验证集（早停）
        valid_loss, valid_acc = evaluate_accuracy_loss(model, valid_loader, DEVICE, loss_function, loss=True)
        # 测试集
        test_acc = evaluate_accuracy_loss(model, test_loader, DEVICE)
        animator.add(epoch + 1, (None, valid_loss, None, valid_acc, test_acc))
        print(f'epoch[{epoch+1}/{EPOCH}] '
              f'train_loss: {train_loss:.3f} valid_loss: {valid_loss:.3f} '
              f'train_acc: {train_acc:.3f} valid_acc: {valid_acc:.3f} test_acc: {test_acc:.3f}')
        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("***** Early stopping *****")
            break
    animator.save(path=f'./model/{func_name}/{model_name} loss_acc.png')
    print(f'{metric[2] * EPOCH / timer.sum():.1f} examples/sec training on {str(DEVICE)}')
    # 学习率变化曲线
    plt.figure()
    plt.title('lr change')
    plt.plot(range(len(lr_list)), lr_list, color='r')
    plt.savefig(f'./model/{func_name}/{model_name} lr_change.png')
    plt.close()

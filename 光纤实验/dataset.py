# -*- coding: utf-8 -*-
# @Time: 2022/11/17
# @Author: Yukai Zeng


import torch
import h5py
import numpy as np
import torch.utils.data as Data
from torch.utils.data.sampler import SubsetRandomSampler


def load_dataset(datafile_name, BATCH_SIZE, func_name, valid_size=None):
    train_dataset = h5py.File(f'./dataset/{datafile_name}_{func_name}_train.h5', "r")
    test_dataset = h5py.File(f'./dataset/{datafile_name}_{func_name}_test.h5', "r")
    x_train = np.array(train_dataset["x_train"])
    y_train = np.array(train_dataset["y_train"])
    x_test = np.array(test_dataset["x_test"])
    y_test = np.array(test_dataset["y_test"])

    # pytorch里有两种数据类型，张量不能反向传播，为了反向传播，需要把张量x转换为variable
    x_train = torch.FloatTensor(x_train.reshape((x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])))
    y_train = torch.LongTensor(y_train)
    x_test = torch.FloatTensor(x_test.reshape((x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])))
    y_test = torch.LongTensor(y_test)
    train_data = Data.TensorDataset(x_train, y_train)
    test_data = Data.TensorDataset(x_test, y_test)
    print("data_train shape: ", x_train.shape, y_train.shape)
    print("data_test shape: ", x_test.shape, y_test.shape)

    # 进行数据封装
    if not valid_size:
        train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
        return train_loader, test_loader
    else:
        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_idx), shuffle=False)
        valid_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(valid_idx), shuffle=False)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
        return train_loader, valid_loader, test_loader
    

if __name__ == '__main__':
    load_dataset('signs_grey', 128, 'morl')



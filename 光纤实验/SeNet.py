#!~/opt/anaconda3/bin/python
# -*- coding: utf-8 -*-
# @Time : 2022/10/4 08:36
# @Author: Mark Zeng
# @Software: PyCharm


import torch
from torch import nn


class SeBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        """ https://blog.csdn.net/weixin_39190382/article/details/117711239
            https://blog.csdn.net/ECHOSON/article/details/121993573?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-121993573-blog-103261226.topnsimilarv1&spm=1001.2101.3001.4242.1&utm_relevant_index=3
            SE注意力机制,输入x。输入输出特征图不变
            1.squeeze: 全局池化 (batch,channel,height,width) -> (batch,channel,1,1) ==> (batch,channel)
            2.excitaton: 全连接or卷积核为1的卷积(batch,channel)->(batch,channel//reduction)-> (batch,channel) ==> (batch,channel,1,1) 输出y
            3.scale: 完成对通道维度上原始特征的标定 y = x*y 输出维度和输入维度相同

        :param channel: 输入特征图的通道数
        :param reduction: 特征图通道的降低倍数
        """
        super(SeBlock, self).__init__()
        # 自适应全局平均池化,即，每个通道进行平均池化，使输出特征图长宽为1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 卷积网络的excitation
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # (batch,channel,height,width)
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1)
        # for i, batch in enumerate(y):
        #     batch_sum = torch.sum(batch)
        #     threshold = 0.05 * batch_sum
        #     y.data[i][torch.where(batch<threshold)] = 0.001 * batch_sum
        return x * y


if __name__ == '__main__':
    input = torch.randn(1, 10, 7, 7)
    se = SeBlock(channel=10, reduction=8)
    output = se(input)
    print(output.shape)
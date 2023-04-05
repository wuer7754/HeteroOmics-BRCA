#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/12/10 15:20
# @Author : xia shufan
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/8 16:20
# @Author  : Xia shufan
# @File    : gcn_model.py
from torch import nn
import torch.nn.functional as F
from gcn_layer import GraphConvolution

class GCN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=None):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(n_in, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_hid) # 使用了一个隐藏层,
        self.gc3 = GraphConvolution(n_hid, n_hid)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        self.dp3 = nn.Dropout(dropout)
        self.fc = nn.Linear(n_hid, n_out) # 这个output真的很重要呢
        self.BatchNorm = nn.BatchNorm1d(n_hid)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, adj):
        # x = self.gc1(input, adj)
        # # x = self.BatchNorm(x) # batch norm  放在激活函数之前先试一试
        # x = F.elu(x)
        # x = self.dp1(x)
        # x = self.gc2(x, adj) # 两个gcn层，两个dp层，激活函数是elu
        # # x = self.BatchNorm(x)
        # x = F.elu(x)
        # x = self.dp2(x)
        # x = self.gc3(x, adj)
        # # x = self.BatchNorm(x)
        # x = F.elu(x)
        # x = self.dp3(x)
        # x = self.fc(x)
        x = self.gc1(input, adj)
        x = F.elu(x)
        x = self.dp1(x)
        x = self.gc2(x, adj)
        x = F.elu(x)
        x = self.dp2(x)
        x = F.elu(x)
        x = self.fc(x)
        # print("gcn_model:")
        # print(x)

        return x
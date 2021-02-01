#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def soft(ratios):
    weights_array = np.array(ratios)
    soft_v = []
    for r in ratios:
        s = np.exp(r) / np.exp(weights_array).sum()
        soft_v.append(s)

    return soft_v


def FedWeightedAvg(w, weights, use_soft=False):
    if use_soft:
        weights = soft(weights)
    # 保证权重和网络选定数目相同
    assert len(weights) == len(w)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = weights[0] * w_avg[k]
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k] + weights[i] * w[i][k]
    return w_avg
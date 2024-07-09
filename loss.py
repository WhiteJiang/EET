# -*- coding: utf-8 -*-
# @Time    : 2024/3/14
# @Author  : White Jiang
import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """
    distillation for cls
    """

    def __init__(self, T=3.0):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        # p_s = F.softmax(y_s / self.T, dim=1)
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)

        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        return loss


class HashDistill(nn.Module):
    def __init__(self):
        super(HashDistill, self).__init__()

    def forward(self, xS, xT):
        loss = (1 - F.cosine_similarity(xS, xT.detach())).mean()
        return loss

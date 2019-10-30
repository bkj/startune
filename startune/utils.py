#!/usr/bin/env python

"""
    utils.py
"""

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seeds(seed):
    _ = np.random.seed(seed )
    _ = torch.manual_seed(seed + 111)
    _ = torch.cuda.manual_seed(seed + 222)
    _ = random.seed(seed + 333)


def sample_gumbel(shape, eps=1e-20):
    U = torch.cuda.FloatTensor(shape).uniform_()
    return -1 * torch.log(-torch.log(U + eps) + eps)

def one_hot_argmax(y):
    shape  = y.size()
    _, ind = y.max(dim=-1)
    one_hot = torch.zeros_like(y).view(-1, shape[-1])
    one_hot.scatter_(1, ind.view(-1, 1), 1)
    one_hot = one_hot.view(*shape)
    return one_hot

def gumbel_softmax(logits, temperature=5):
    y = logits + sample_gumbel(logits.size())
    y = F.softmax(y / temperature, dim=-1)
    
    y_hard = one_hot_argmax(y)
    return (y_hard - y).detach() + y

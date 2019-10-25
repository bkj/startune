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


def adjust_learning_rate_net(optimizer, epoch, args):
    
    if epoch >= args.step3:
        lr = args.lr * 0.001
    if epoch >= args.step2:
        lr = args.lr * 0.01
    if epoch >= args.step1:
        lr = args.lr * 0.1
    else:
        lr = args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_agent(optimizer, epoch, args):
    
    if epoch >= args.step3:
        lr = args.lr_agent * 0.001
    if epoch >= args.step2:
        lr = args.lr_agent * 0.01
    if epoch >= args.step1:
        lr = args.lr_agent * 0.1
    else:
        lr = args.lr_agent
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def sample_gumbel(shape, eps=1e-20):
    U = torch.cuda.FloatTensor(shape).uniform_()
    return -1 * torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=5):
    y      = gumbel_softmax_sample(logits, temperature)
    
    shape  = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

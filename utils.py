#!/usr/bin/env python

"""
    utils.py
"""

import random
import numpy as np

import torch
import torch.nn as nn

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

#!/usr/bin/env python

"""
    simple_main.py
"""

import os
import sys
import json
import argparse
import collections
import numpy as np
from time import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

from startune.data import get_data
from startune.models import ResNet, TwoPathModel, StarTuneWrapper

from startune.utils import set_seeds

torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True

torch.set_num_threads(1)

# --
# Helpers

weight_decays = {
    "aircraft"      : 0.0005,
    "cifar100"      : 0.0,
    "daimlerpedcls" : 0.0005,
    "dtd"           : 0.0,
    "gtsrb"         : 0.0,
    "omniglot"      : 0.0005,
    "svhn"          : 0.0,
    "ucf101"        : 0.0005,
    "vgg-flowers"   : 0.0001,
    # "imagenet12"    : 0.0001,
}

def do_epoch(model, loader, model_opt=None, agent_opt=None, straight=False):
    total_seen, total_loss, total_correct = 0, 0, 0
    
    for i, (x, y) in enumerate(tqdm(loader, total=len(loader))):
        x, y = x.cuda(), y.cuda()
        
        out  = model(x, straight=straight)
        
        loss = F.cross_entropy(out, y)
        
        preds   = torch.argmax(out.data, dim=-1)
        correct = int((preds == y).sum())
        
        total_seen    += int(y.shape[0])
        total_loss    += float(loss)
        total_correct += correct
        
        if model.training:
            _ = model_opt.zero_grad()
            _ = agent_opt.zero_grad()
            _ = loss.backward()
            _ = model_opt.step()
            _ = agent_opt.step()
        
    acc  = total_correct / total_seen
    loss = total_loss / total_seen
    
    return acc, loss

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SpotTune')
    
    parser.add_argument('--inpath',     type=str, default='./data/decathlon-1.0/')
    parser.add_argument('--outpath',    type=str, default='model')
    parser.add_argument('--model-path', type=str, default='models/ResNet.t7')
    parser.add_argument('--dataset',    type=str, default='aircraft')
    
    parser.add_argument('--epochs',   type=int,   default=110)
    parser.add_argument('--lr',       type=float, default=0.1)
    parser.add_argument('--lr-agent', type=float, default=0.01)
    
    parser.add_argument('--valid-interval', type=int, default=1)
    
    parser.add_argument('--train-on-valid', action="store_true")
    parser.add_argument('--straight',       action="store_true")
    
    parser.add_argument('--lr-sched',      default='step', type=str)
    parser.add_argument('--lr-milestones', default='40,60,80', type=str)
    
    parser.add_argument('--seed', default=123, type=int, help='seed')
    
    return parser.parse_args()

# --
# Run

args = parse_args()
open(args.outpath + '.json', 'w').write(json.dumps(vars(args)))

set_seeds(args.seed)

# --
# Data

train_loader, valid_loader, n_class = get_data(
    root=args.inpath,
    dataset=args.dataset,
    shuffle_train=True,
    train_on_valid=args.train_on_valid
)

# --
# Models

pretrained = torch.load(args.model_path)['net']

model = StarTuneWrapper(
    twopath=TwoPathModel(model=pretrained, n_class=n_class),
    agent=nn.Sequential(
        ResNet(nblocks=[1, 1, 1]),
        nn.Linear(pretrained.out_channels, 24) # !! I think this could be 12 instead of 24
    ),
)

_ = model.cuda()

# >>
# model = torch.nn.DataParallel(model)
# <<

model_params = [p for k,p in model.named_parameters() if (('twopath.' in k) and (p.requires_grad))]
agent_params = [p for k,p in model.named_parameters() if (('agent.' in k) and (p.requires_grad))]

model_opt = torch.optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=weight_decays[args.dataset])
agent_opt = torch.optim.SGD(agent_params, lr=args.lr_agent, momentum=0.9, weight_decay=0.001)

if args.lr_sched == 'step':
    model_sched = MultiStepLR(model_opt, milestones=eval(args.lr_milestones), gamma=0.1)
    agent_sched = MultiStepLR(agent_opt, milestones=eval(args.lr_milestones), gamma=0.1)
elif args.lr_sched == 'cosine':
    model_sched = CosineAnnealingLR(model_opt, T_max=args.epochs)
    agent_sched = CosineAnnealingLR(agent_opt, T_max=args.epochs)
elif args.lr_sched == 'constant':
    model_sched = None
    agent_sched = None
else:
    raise Exception()

# --
# Train

t = time()

valid_acc, valid_loss = -1, -1

for epoch in range(args.epochs):
    
    _ = model.train()
    train_acc, train_loss = do_epoch(model, train_loader, model_opt=model_opt, agent_opt=agent_opt, straight=args.straight)
    
    if (not epoch % args.valid_interval) or (epoch == args.epochs - 1):
        _ = model.eval()
        valid_acc, valid_loss = do_epoch(model, valid_loader, straight=args.straight)
    
    print(json.dumps({
        "dataset"    : args.dataset,
        "epoch"      : epoch,
        "train_acc"  : float(train_acc),
        "train_loss" : float(train_loss),
        "valid_acc"  : float(valid_acc),
        "valid_loss" : float(valid_loss),
        "elapsed"    : float(time() - t),
    }))
    sys.stdout.flush()
    
    model_sched.step()
    agent_sched.step()


print(f'startune.main: saving to {args.outpath}', file=sys.stderr)

# if hasattr(model, 'module'):
#     model = model.module

torch.save(model, args.outpath)
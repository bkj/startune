#!/usr/bin/env python

"""
    startune/predict.py
"""

from rsub import *
from matplotlib import pyplot as plt

import os
import sys
import json
import argparse
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

from startune.data import get_test_data, get_data

from startune.utils import set_seeds, gumbel_softmax, one_hot_argmax

torch.backends.cudnn.deterministic = True

# --
# Helpers

def path_tail(p, k=1):
    return '/'.join(p.split('/')[-k:])


def get_dir2lab(ann):
    id2dir = pd.DataFrame([{
        "id"  : a['id'],
        "dir" : os.path.basename(os.path.dirname(a['file_name'])),
    } for a in ann['images']])
    
    id2lab = pd.DataFrame([{
        "id"  : a['id'],
        "lab" : a['category_id'],
    } for a in ann['annotations']])
    
    dir2lab = pd.merge(id2dir, id2lab)[['dir', 'lab']]
    dir2lab = {row.dir:row.lab for _, row in dir2lab.iterrows()}
    return dir2lab


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

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--inpath',     type=str, default='./data/decathlon-1.0/')
    parser.add_argument('--dataset',    type=str, default='vgg-flowers')
    parser.add_argument('--model',      type=str, default='models/exp1/spottune/vgg-flowers.pth')
    parser.add_argument('--mode',       type=str, default='test')
    
    parser.add_argument('--seed', default=123, type=int)
    
    return parser.parse_args()


args = parse_args()
set_seeds(args.seed)


def predict(model, loader, straight=False, greedy=False):
    _ = model.eval()
    
    all_preds = []
    
    with torch.no_grad():
        for i, (x, _) in enumerate(tqdm(loader, total=len(loader))):
            x = x.cuda()
            
            preds = model(x, straight=straight, greedy=greedy).argmax(dim=-1)
            preds = preds.detach().cpu()
            all_preds.append(preds)
    
    return torch.cat(all_preds)


train_loader, valid_loader, _ = get_data(
    root=args.inpath,
    dataset=args.dataset,
    shuffle_train=False,
    train_on_valid=True,
    num_workers=0
)

y  = torch.cat([y for _, y in valid_loader])

# --
# Models

model = torch.load(args.model, map_location=lambda *x: x[0])
model = model.eval().cuda()

all_accs += [(y == predict(model, loader=valid_loader)).float().mean() for _ in range(10)]
torch.Tensor(all_accs).mean()

# --
# Show routing statistics

x, _ = valid_loader[0]
x = x.cuda()

# Greedy policy
gpolicy = model(x, greedy=True, return_policy=True)
gpolicy = gpolicy.detach().cpu()

# Sampled policies
sampled_policies = []
for _ in trange(100):
    policy = model(x, return_policy=True)
    sampled_policies.append(policy.detach().cpu())

sampled_policies = torch.stack(sampled_policies)

mean_gpolicy        = gpolicy.mean(dim=0)
mean_sampled_policy = sampled_policies.mean(dim=(0, 1))

_ = plt.plot(mean_gpolicy, marker='o', label='greedy')
_ = plt.plot(mean_sampled_policy, marker='o', label='sampled')
_ = plt.ylim(-0.1, 1.1)
_ = plt.grid(alpha=0.25)
_ = plt.legend()
show_plot()

# --
# Greedy routing

y_gpred = predict(model, loader=valid_loader, greedy=True)
(y == y_gpred).float().mean()



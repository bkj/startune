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
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from startune.data import get_test_data, get_data
from startune.models import SimpleResNet, SimpleStarNet

from startune.utils import (
    set_seeds, gumbel_softmax
)

raise Exception() # What to do here?
# torch.backends.cudnn.deterministic = True

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

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--inpath',     type=str, default='./data/decathlon-1.0/')
    parser.add_argument('--dataset',    type=str, default='aircraft')
    parser.add_argument('--outpath',    type=str, default='predictions.json')
    parser.add_argument('--model-path', type=str, default='models/aircraft.pth')
    
    parser.add_argument('--seed', default=123, type=int, help='seed')
    
    return parser.parse_args()


args = parse_args()
set_seeds(args.seed)


def predict(model, agent, loader):
    _ = model.eval()
    _ = agent.eval()
    
    all_preds = []
    
    with torch.no_grad():
        for i, (x, _) in enumerate(tqdm(loader, total=len(loader))):
            x = x.cuda()
            
            probs  = agent(x)
            action = gumbel_softmax(probs.view(probs.shape[0], -1, 2))
            policy = action[:,:,1]
            
            out   = model(x, policy)
            preds = torch.argmax(out.data, dim=-1)
            all_preds.append(preds.detach().cpu())
    
    return torch.cat(all_preds)

# --
# Load annotation information

train_ann_path = os.path.join(args.inpath, 'annotations', f'{args.dataset}_train.json')
test_ann_path  = os.path.join(args.inpath, 'annotations', f'{args.dataset}_test_stripped.json')

train_ann = json.load(open(train_ann_path))
test_ann  = json.load(open(test_ann_path))

idx2cls = {idx:c['id'] for idx, c in enumerate(train_ann['categories'])}
img2id  = {os.path.basename(a['file_name']):a['id'] for a in test_ann['images']}

# --
# Data

test_loader = get_test_data(
    root=args.inpath,
    dataset=args.dataset,
)

n_class = len(test_loader.dataset.classes)

assert len(test_loader.dataset) == len(img2id), 'len(test_loader.dataset) != len(img2id)'

# --
# Models

checkpoint = torch.load(args.model_path, map_location=lambda *x: x[0])

model = checkpoint['model']
agent = checkpoint['agent']

_ = model.cuda().eval()
_ = agent.cuda().eval()

all_preds = predict(model, agent, loader=test_loader)
filenames = [os.path.basename(p) for p, _ in test_loader.dataset.imgs]

assert len(filenames) == len(all_preds), 'len(filenames) != len(all_preds)'

for filename, pred in zip(filenames, all_preds):
    print(json.dumps({
        "image_id"    : img2id[filename],
        "category_id" : idx2cls[int(pred)],
    }))


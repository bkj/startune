#!/usr/bin/env python

"""
    startune/predict.py
    
    (Ugly) code to make leaderboard predictions
"""

import os
import sys
import json
import argparse
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from startune.data import get_test_data, get_data

from startune.utils import (
    set_seeds, gumbel_softmax
)

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
    parser.add_argument('--dataset',    type=str, default='aircraft')
    parser.add_argument('--model',      type=str, default='models/aircraft.pth')
    parser.add_argument('--mode',       type=str, default='test')
    
    parser.add_argument('--straight', action="store_true")
    parser.add_argument('--seed', default=123, type=int)
    
    return parser.parse_args()


args = parse_args()
set_seeds(args.seed)


def predict(model, loader, straight):
    _ = model.eval()
    
    all_preds = []
    
    with torch.no_grad():
        for i, (x, _) in enumerate(tqdm(loader, total=len(loader))):
            x = x.cuda()
            
            preds = model(x, straight=straight).argmax(dim=-1)
            preds = preds.detach().cpu()
            all_preds.append(preds)
    
    return torch.cat(all_preds)

# --
# Load annotation information

train_ann_path  = os.path.join(args.inpath, 'annotations', f'{args.dataset}_train.json')
train_ann       = json.load(open(train_ann_path))

dir2lab = get_dir2lab(train_ann)
dir2lab = list(dir2lab.values())

if args.mode == 'test':
    target_ann_path = os.path.join(args.inpath, 'annotations', f'{args.dataset}_test_stripped.json')
    target_ann      = json.load(open(target_ann_path))
    img2id          = {path_tail(a['file_name'], k=1):a['id'] for a in target_ann['images']}
elif args.mode == 'valid':
    target_ann_path = os.path.join(args.inpath, 'annotations', f'{args.dataset}_val.json')
    target_ann      = json.load(open(target_ann_path))
    img2id          = {a['file_name']:a['id'] for a in target_ann['images']}

# --
# Data

if args.mode == 'test':
    loader = get_test_data(
        root=args.inpath,
        dataset=args.dataset,
    )
elif args.mode == 'valid':
    _, loader = get_data(
        root=args.inpath,
        dataset=args.dataset,
        shuffle_train=False,
        train_on_valid=True,
        num_workers=0
    )

if args.mode == 'test':
    filenames = [path_tail(p, k=1) for p, _ in loader.dataset.imgs]
elif args.mode == 'valid':
    filenames = [path_tail(p, k=5) for p, _ in loader.dataset.imgs]

# --
# Models

model = torch.load(args.model, map_location=lambda *x: x[0])
# >>
model = nn.DataParallel(model)
# <<

_ = model.eval().cuda()

all_preds = predict(model, loader=loader, straight=args.straight)

for filename, pred in zip(filenames, all_preds):
    print(json.dumps({
        "image_id"    : img2id[filename],
        "category_id" : dir2lab[int(pred)],
    }))


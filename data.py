#!/usr/bin/env python

"""
    data.py
"""

import os
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

def get_transform(dataset, means, stds):
    if dataset in ['gtsrb', 'omniglot','svhn']: 
        transform_train = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(72),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
        transform_valid = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(72),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    elif dataset in ['daimlerpedcls']:
        transform_train = transforms.Compose([
            transforms.Resize(72),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
        transform_valid = transforms.Compose([
            transforms.Resize(72),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(72),
            transforms.RandomCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
        transform_valid = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(72),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    
    return transform_train, transform_valid

def prepare_data_loaders(dataset_names, data_dir, shuffle_train=True, batch_size=128, num_workers=8):
    
    dict_mean_std = pickle.load(open(data_dir + 'decathlon_mean_std.pickle'))
    
    dataloaders = {}
    for dataset in dataset_names:
        
        transform_train, transform_valid = get_transform(
            dataset = dataset, 
            means   = dict_mean_std[dataset + 'mean'],
            stds    = dict_mean_std[dataset + 'std'],
        )
        
        dataloaders[dataset] = {
            'train' : DataLoader(
                ImageFolder(
                    root      = os.path.join(data_dir, 'data', dataset, 'train'),
                    transform = transform_train
                ),
                shuffle     = shuffle_train,
                batch_size  = batch_size,
                num_workers = num_workers,
            ),
            'valid' : DataLoader(
                ImageFolder(
                    root      = os.path.join(data_dir, 'data', dataset, 'val'),
                    transform = transform_valid
                ),
                shuffle     = False, 
                batch_size  = batch_size,
                num_workers = num_workers,
            )
        }
    
    return dataloaders



#!/usr/bin/env python

"""
    data.py
"""

import os
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader, ConcatDataset

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

def get_data(root, dataset, train_on_valid=False, shuffle_train=True, batch_size=128, num_workers=8):
    
    dict_mean_std = pickle.load(open(root + 'decathlon_mean_std.pickle', 'rb'), encoding='latin1')
        
    transform_train, transform_valid = get_transform(
        dataset = dataset, 
        means   = dict_mean_std[dataset + 'mean'],
        stds    = dict_mean_std[dataset + 'std'],
    )
    
    valid_path = os.path.join(root, 'data', dataset, 'val')
    train_path = os.path.join(root, 'data', dataset, 'train')
    
    valid_dataset = ImageFolder(root=valid_path, transform=transform_valid)
    
    if train_on_valid:
        train_dataset = ConcatDataset([
            ImageFolder(root=train_path, transform=transform_train),
            ImageFolder(root=valid_path, transform=transform_train),
        ])
    else:
        train_dataset = ImageFolder(root=train_path, transform=transform_train)
    
    train_loader = DataLoader(
        train_dataset,
        shuffle     = shuffle_train,
        batch_size  = batch_size,
        num_workers = num_workers,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        shuffle     = False, 
        batch_size  = batch_size,
        num_workers = num_workers,
    )
    
    return train_loader, valid_loader



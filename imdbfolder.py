import os
import pickle

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import numpy as np

def prepare_data_loaders(dataset_names, data_dir, shuffle_train=True, batch_size=128, num_workers=8):
    dataloaders = {}
    
    dict_mean_std = pickle.load(open(data_dir + 'decathlon_mean_std.pickle'))
    
    for dataset in dataset_names:
        
        means = dict_mean_std[dataset + 'mean']
        stds  = dict_mean_std[dataset + 'std']
        
        if dataset in ['gtsrb', 'omniglot','svhn']: 
            transform_train = transforms.Compose([
                transforms.Resize(72),
                transforms.CenterCrop(72),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
            transform_test = transforms.Compose([
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
            transform_test = transforms.Compose([
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
            transform_test = transforms.Compose([
                transforms.Resize(72),
                transforms.CenterCrop(72),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
        
        dataset_train = ImageFolder(os.path.join(data_dir, 'data', dataset, 'train'), transform=transform_train)
        dataset_valid = ImageFolder(os.path.join(data_dir, 'data', dataset, 'val'), transform=transform_test)
        
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        dataloaders[dataset] = {
            "train" : dataloader_train,
            "valid" :dataloader_valid,
        }
    
    return dataloaders



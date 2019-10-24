#!/usr/bin/env python

"""
    main2.py
"""

import os
import sys
import time
import json
import random
import argparse
import collections
import numpy as np

import torch

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import imdbfolder2
from models2 import ResNet2, STResNet2
from utils import *
from gumbel_softmax import *

torch.backends.cudnn.deterministic = True

# --
# Helpers

def set_seeds(seed):
    _ = np.random.seed(seed )
    _ = torch.manual_seed(seed + 111)
    _ = torch.cuda.manual_seed(seed + 222)
    _ = random.seed(seed + 333)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SpotTune')
    
    parser.add_argument('--nb_epochs', default=110, type=int, help='nb epochs')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate of net')
    parser.add_argument('--lr_agent', default=0.01, type=float, help='initial learning rate of agent')
    
    parser.add_argument('--datadir', default='./data/decathlon-1.0/', help='folder containing data folder')
    parser.add_argument('--imdbdir', default='./data/decathlon-1.0/annotations/', help='annotation folder')
    parser.add_argument('--ckpdir', default='./cv/', help='folder saving checkpoint')
    
    parser.add_argument('--seed', default=0, type=int, help='seed')
    
    parser.add_argument('--step1', default=40, type=int, help='nb epochs before first lr decrease')
    parser.add_argument('--step2', default=60, type=int, help='nb epochs before second lr decrease')
    parser.add_argument('--step3', default=80, type=int, help='nb epochs before third lr decrease')
    
    return parser.parse_args()

args = parse_args()
set_seeds(args.seed)

weight_decays = [
    ("aircraft", 0.0005),
    # ("cifar100", 0.0),
    # ("daimlerpedcls", 0.0005),
    # ("dtd", 0.0),
    # ("gtsrb", 0.0),
    # ("omniglot", 0.0005),
    # ("svhn", 0.0),
    # ("ucf101", 0.0005),
    # ("vgg-flowers", 0.0001),
    # ("imagenet12", 0.0001)
]

datasets = [
    ("aircraft", 0),
    # ("cifar100", 1),
    # ("daimlerpedcls", 2),
    # ("dtd", 3),
    # ("gtsrb", 4),
    # ("omniglot", 5),
    # ("svhn", 6),
    # ("ucf101", 7),
    # ("vgg-flowers", 8)
]

datasets      = collections.OrderedDict(datasets)
weight_decays = collections.OrderedDict(weight_decays)

with open(args.ckpdir + '/weight_decays.json', 'w') as fp:
    json.dump(weight_decays, fp)

def train(dataset, poch, train_loader, net, agent, net_optimizer, agent_optimizer):
    _ = net.train()
    _ = agent.train()
    
    tasks_top1   = AverageMeter()
    tasks_losses = AverageMeter()
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(async=True), labels.cuda(async=True)
        images, labels = Variable(images), Variable(labels)
        
        probs  = agent(images)
        action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
        policy = action[:,:,1]
        
        outputs = net.forward(images, policy)
        _, predicted = torch.max(outputs.data, 1)
        
        correct = predicted.eq(labels.data).cpu().sum()
        tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))
        
        loss = F.cross_entropy(outputs, labels)
        tasks_losses.update(loss.item(), labels.size(0))
        
        # Step
        _ = net_optimizer.zero_grad()
        _ = agent_optimizer.zero_grad()
        _ = loss.backward()
        _ = net_optimizer.step()
        _ = agent_optimizer.step()
        
    return tasks_top1.avg, tasks_losses.avg


def test(epoch, val_loader, net, agent, dataset):
    _ = net.eval()
    _ = agent.eval()
    
    tasks_top1   = AverageMeter()
    tasks_losses = AverageMeter()
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.cuda(async=True), labels.cuda(async=True)
            images, labels = Variable(images), Variable(labels)
            
            probs  = agent(images)
            action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
            policy = action[:,:,1]
            
            outputs = net.forward(images, policy)
            _, predicted = torch.max(outputs.data, 1)
            
            correct = predicted.eq(labels.data).cpu().sum()
            tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))
            
            loss = F.cross_entropy(outputs, labels)
            tasks_losses.update(loss.item(), labels.size(0))
            
    return tasks_top1.avg, tasks_losses.avg


dataset = list(datasets.keys())[0]

# >>
train_loaders, valid_loaders, num_classes = imdbfolder2.prepare_data_loaders(datasets.keys(), args.datadir, args.imdbdir, True)
train_loader = train_loaders[datasets[dataset]]
valid_loader = valid_loaders[datasets[dataset]]
num_class    = num_classes[datasets[dataset]]
# --
# !! This should be the same, but I think order is slightly different
# dataloaders = \
#     imdbfolder.prepare_data_loaders(datasets.keys(), args.datadir, shuffle_train=True)
# train_loader = dataloaders[dataset]['valid']
# valid_loader = dataloaders[dataset]['valid']
# num_class    = len(train_loader.dataset.classes)
# <<

net   = STResNet2(torch.load('tmp2.t7')['net'])
agent = nn.Sequential(
    ResNet2(nblocks=[1, 1, 1]),
    nn.Linear(256, 24)          # !! I think this is the wrong dimensionality
)

_ = net.cuda()
_ = agent.cuda()

net_params   = filter(lambda p: p.requires_grad, net.parameters())
agent_params = agent.parameters()

net_optimizer   = optim.SGD(net_params, lr= args.lr, momentum=0.9, weight_decay=weight_decays[dataset])
agent_optimizer = optim.SGD(agent_params, lr= args.lr_agent, momentum= 0.9, weight_decay= 0.001)

for epoch in range(args.nb_epochs):
    adjust_learning_rate_net(net_optimizer, epoch, args)
    adjust_learning_rate_agent(agent_optimizer, epoch, args)
    
    train_acc, train_loss = train(dataset, epoch, train_loader, net, agent, net_optimizer, agent_optimizer)
    test_acc, test_loss   = test(epoch, valid_loader, net, agent, dataset)
    
    print(json.dumps({
        "dataset"    : dataset,
        "train_acc"  : float(train_acc),
        "train_loss" : float(train_loss),
        "test_acc"   : float(test_acc),
        "test_loss"  : float(test_loss),
    }))
    sys.stdout.flush()

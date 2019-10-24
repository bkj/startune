#!/usr/bin/env python

"""
    main2.py
"""

import os
import sys
import json
import argparse
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from data import prepare_data_loaders
from simple_models import SimpleResNet, SimpleStarNet

from gumbel_softmax import gumbel_softmax
from utils import AverageMeter, adjust_learning_rate_net, adjust_learning_rate_agent, set_seeds

torch.backends.cudnn.deterministic = True

# --
# Helpers

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

def train(model, agent, train_loader, model_opt, agent_opt):
    _ = model.train()
    _ = agent.train()
    
    tasks_top1   = AverageMeter()
    tasks_losses = AverageMeter()
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(async=True), labels.cuda(async=True)
        images, labels = Variable(images), Variable(labels)
        
        probs  = agent(images)
        action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
        policy = action[:,:,1]
        
        outputs = model(images, policy)
        _, predicted = torch.max(outputs.data, 1)
        
        correct = predicted.eq(labels.data).cpu().sum()
        tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))
        
        loss = F.cross_entropy(outputs, labels)
        tasks_losses.update(loss.item(), labels.size(0))
        
        # Step
        _ = model_opt.zero_grad()
        _ = agent_opt.zero_grad()
        _ = loss.backward()
        _ = model_opt.step()
        _ = agent_opt.step()
        
    return tasks_top1.avg, tasks_losses.avg


def valid(model, agent, valid_loader):
    _ = model.eval()
    _ = agent.eval()
    
    tasks_top1   = AverageMeter()
    tasks_losses = AverageMeter()
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(valid_loader):
            images, labels = images.cuda(async=True), labels.cuda(async=True)
            images, labels = Variable(images), Variable(labels)
            
            probs  = agent(images)
            action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
            policy = action[:,:,1]
            
            outputs = model(images, policy)
            _, predicted = torch.max(outputs.data, 1)
            
            correct = predicted.eq(labels.data).cpu().sum()
            tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))
            
            loss = F.cross_entropy(outputs, labels)
            tasks_losses.update(loss.item(), labels.size(0))
            
    return tasks_top1.avg, tasks_losses.avg


# --
# Data

dataset = list(datasets.keys())[0]

dataloaders  = prepare_data_loaders(datasets.keys(), args.datadir, shuffle_train=True)
train_loader = dataloaders[dataset]['train']
valid_loader = dataloaders[dataset]['valid']
num_class    = len(train_loader.dataset.classes)

# --
# Models

model = STResNet2(torch.load('tmp2.t7')['net'])
agent = nn.Sequential(
    ResNet2(nblocks=[1, 1, 1]),
    nn.Linear(256, 24)          # !! I think this is twice the necessary dim
)

_ = model.cuda()
_ = agent.cuda()

model_params = filter(lambda p: p.requires_grad, model.parameters())
agent_params = agent.parameters()

model_opt = torch.optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=weight_decays[dataset])
agent_opt = torch.optim.SGD(agent_params, lr=args.lr_agent, momentum=0.9, weight_decay=0.001)

# --
# Train

for epoch in range(args.nb_epochs):
    adjust_learning_rate_net(model_opt, epoch, args)
    adjust_learning_rate_agent(agent_opt, epoch, args)
    
    train_acc, train_loss = train(model, agent, train_loader, model_opt, agent_opt)
    valid_acc, valid_loss = valid(model, agent, valid_loader)
    
    print(json.dumps({
        "dataset"    : dataset,
        "epoch"      : epoch,
        "train_acc"  : float(train_acc),
        "train_loss" : float(train_loss),
        "valid_acc"  : float(valid_acc),
        "valid_loss" : float(valid_loss),
    }))
    sys.stdout.flush()

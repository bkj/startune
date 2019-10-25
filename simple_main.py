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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from data import get_data
from simple_models import SimpleResNet, SimpleStarNet

from gumbel_softmax import gumbel_softmax
from utils import AverageMeter, adjust_learning_rate_net, adjust_learning_rate_agent, set_seeds

torch.backends.cudnn.deterministic = True

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
    "imagenet12"    : 0.0001,
}

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SpotTune')
    
    parser.add_argument('--inpath',     type=str, default='./data/decathlon-1.0/')
    parser.add_argument('--outpath',    type=str, default='model')
    parser.add_argument('--model-path', type=str, default='models/SimpleResNet.t7')
    parser.add_argument('--dataset',    type=str, default='aircraft')
    
    parser.add_argument('--epochs',   type=int,   default=110)
    parser.add_argument('--lr',       type=float, default=0.1)
    parser.add_argument('--lr-agent', type=float, default=0.01)
    
    parser.add_argument('--train-on-valid', action="store_true")
    
    parser.add_argument('--step1', default=40, type=int, help='epochs before first lr decrease')
    parser.add_argument('--step2', default=60, type=int, help='epochs before second lr decrease')
    parser.add_argument('--step3', default=80, type=int, help='epochs before third lr decrease')
    
    parser.add_argument('--seed', default=123, type=int, help='seed')
    
    return parser.parse_args()


args = parse_args()
set_seeds(args.seed)

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

train_loader, valid_loader = get_data(
    root=args.inpath, dataset=args.dataset, shuffle_train=True, train_on_valid=args.train_on_valid)

num_class = len(train_loader.dataset.classes)

# --
# Models

model = SimpleStarNet(torch.load(args.model_path)['net'])
agent = nn.Sequential(
    SimpleResNet(nblocks=[1, 1, 1]),
    nn.Linear(256, 24)               # !! I think this is twice the necessary dim
)

_ = model.cuda()
_ = agent.cuda()

model_params = filter(lambda p: p.requires_grad, model.parameters())
agent_params = agent.parameters()

model_opt = torch.optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=weight_decays[args.dataset])
agent_opt = torch.optim.SGD(agent_params, lr=args.lr_agent, momentum=0.9, weight_decay=0.001)

# --
# Train

torch.save(model, args.outpath)

for epoch in range(args.epochs):
    adjust_learning_rate_net(model_opt, epoch, args)
    adjust_learning_rate_agent(agent_opt, epoch, args)
    
    train_acc, train_loss = train(model, agent, train_loader, model_opt, agent_opt)
    valid_acc, valid_loss = valid(model, agent, valid_loader)
    
    print(json.dumps({
        "dataset"    : args.dataset,
        "epoch"      : epoch,
        "train_acc"  : float(train_acc),
        "train_loss" : float(train_loss),
        "valid_acc"  : float(valid_acc),
        "valid_loss" : float(valid_loss),
    }))
    sys.stdout.flush()

torch.save(model, args.outpath)
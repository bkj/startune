#!/usr/bin/env python

"""
    main.py
"""

import os
import json
import argparse
import numpy as np
import collections
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import imdbfolder as imdbfolder
from spottune_models import *
import models
import agent_net
from utils import *
from gumbel_softmax import *

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--nb_epochs', default=110, type=int, help='nb epochs')
    parser.add_argument('--lr',        default=0.1, type=float, help='initial learning rate of net')
    parser.add_argument('--lr_agent',  default=0.01, type=float, help='initial learning rate of agent')
    
    parser.add_argument('--datadir', default='./data/decathlon-1.0/', help='folder containing data folder')
    parser.add_argument('--imdbdir', default='./data/decathlon-1.0/annotations/', help='annotation folder')
    parser.add_argument('--ckpdir',  default='./cv/', help='folder saving checkpoint')
    
    parser.add_argument('--seed',  default=0, type=int, help='seed')
    
    parser.add_argument('--step1', default=40, type=int, help='nb epochs before first lr decrease')
    parser.add_argument('--step2', default=60, type=int, help='nb epochs before second lr decrease')
    parser.add_argument('--step3', default=80, type=int, help='nb epochs before third lr decrease')
    
    return parser.parse_args()

args = parse_args()

_ = np.random.seed(args.seed)
_ = torch.manual_seed(args.seed + 111)
_ = torch.cuda.manual_seed(args.seed + 222)

# --
# Params

weight_decays = [
    ("aircraft",      0.0005),
    ("cifar100",      0.0),
    ("daimlerpedcls", 0.0005),
    ("dtd",           0.0),
    ("gtsrb",         0.0),
    ("omniglot",      0.0005),
    ("svhn",          0.0),
    ("ucf101",        0.0005),
    ("vgg-flowers",   0.0001),
    ("imagenet12",    0.0001),
]

datasets = [
    ("aircraft",       0),
    ("cifar100",       1),
    ("daimlerpedcls",  2),
    ("dtd",            3),
    ("gtsrb",          4),
    ("omniglot",       5),
    ("svhn",           6),
    ("ucf101",         7),
    ("vgg-flowers",    8),
]

datasets      = collections.OrderedDict(datasets)
weight_decays = collections.OrderedDict(weight_decays)

with open(args.ckpdir + '/weight_decays.json', 'w') as fp:
    json.dump(weight_decays, fp)

# --
# Train helper

def train(dataset, poch, train_loader, net, agent, net_opt, agent_opt):
    _ = net.train()
    _ = agent.train()
    
    total_step   = len(train_loader)
    tasks_top1   = AverageMeter()
    tasks_losses = AverageMeter()
    
    for i, (images, labels) in enumerate(train_loader):
        
        images, labels = Variable(images.cuda(async=True)), Variable(labels.cuda(async=True))
        
        probs  = agent(images)
        action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
        policy = action[:,:,1]
        
        outputs = net.forward(images, policy)
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(labels.data).cpu().sum()
        tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))
        
        loss = criterion(outputs, labels)
        tasks_losses.update(loss.item(), labels.size(0))
        
        if i % 50 == 0:
            print({
                "epoch"   : epoch,
                "step"    : i + 1,
                "loss"    : tasks_losses.val,
                "acc"     : tasks_top1.val,
                "avg_acc" : tasks_top1.avg,
            })
            sys.stdout.flush()
       
        net_opt.zero_grad()
        agent_opt.zero_grad()
        
        loss.backward()
        net_opt.step()
        agent_opt.step()
    
    return tasks_top1.avg , tasks_losses.avg

def test(epoch, val_loader, net, agent, dataset):
    net.eval()
    agent.eval()

    tasks_top1 = AverageMeter()
    tasks_losses = AverageMeter() 

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if use_cuda:
                images, labels = images.cuda(async=True), labels.cuda(async=True)
            images, labels = Variable(images), Variable(labels)

       	    probs = agent(images)
            action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
            policy = action[:,:,1]
            outputs = net.forward(images, policy)

            _, predicted = torch.max(outputs.data, 1)
            correct = predicted.eq(labels.data).cpu().sum()
            tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))
        
            # Loss
            loss = criterion(outputs, labels)
            tasks_losses.update(loss.item(), labels.size(0))           

    print "test accuracy"
    print ("Epoch [{}/{}], Loss: {:.4f}, Acc Val: {:.4f}, Acc Avg: {:.4f}"
        .format(epoch+1, args.nb_epochs, tasks_losses.avg, tasks_top1.val, tasks_top1.avg))

    return tasks_top1.avg, tasks_losses.avg

def load_weights_to_flatresnet(source, net, num_class, dataset):
    checkpoint = torch.load(source)
    net_old = checkpoint['net']

    store_data = []
    t = 0
    for name, m in net_old.named_modules():
        if isinstance(m, nn.Conv2d):
            store_data.append(m.weight.data)
            t += 1

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and 'parallel_blocks' not in name:
            m.weight.data = torch.nn.Parameter(store_data[element].clone())
            element += 1

    element = 1
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and 'parallel_blocks' in name:
            m.weight.data = torch.nn.Parameter(store_data[element].clone())
            element += 1

    store_data = []
    store_data_bias = []
    store_data_rm = []
    store_data_rv = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            store_data.append(m.weight.data)
            store_data_bias.append(m.bias.data)
            store_data_rm.append(m.running_mean)
            store_data_rv.append(m.running_var)

    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d) and 'parallel_block' not in name:
            	m.weight.data = torch.nn.Parameter(store_data[element].clone())
                m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
                m.running_var = store_data_rv[element].clone()
                m.running_mean = store_data_rm[element].clone()
                element += 1

    element = 1
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d) and 'parallel_block' in name:
                m.weight.data = torch.nn.Parameter(store_data[element].clone())
                m.bias.data = torch.nn.Parameter(store_data_bias[element].clone())
                m.running_var = store_data_rv[element].clone()
                m.running_mean = store_data_rm[element].clone()
                element += 1
    
    del net_old
    return net

def get_model(model, num_class, dataset = None):
    if model == 'resnet26':
        rnet = resnet26(num_class)
        if dataset is not None:
            if dataset == 'imagenet12':
            	source = './resnet26_pretrained.t7'
	    else:
            	source = './cv/' + dataset + '/' + dataset + '.t7'
        rnet = load_weights_to_flatresnet(source, rnet, num_class, dataset)
    return rnet

#####################################
# Prepare data loaders
train_loaders, val_loaders, num_classes = imdbfolder.prepare_data_loaders(datasets.keys(), args.datadir, args.imdbdir, True)
criterion = nn.CrossEntropyLoss()

for i, dataset in enumerate(datasets.keys()):
    print dataset 
    
    pretrained_model_dir = args.ckpdir + dataset
    
    os.mkdir(pretrained_model_dir, exist_ok=True)
    
    results = np.zeros((4, args.nb_epochs, len(num_classes)))
    
    json.dump(vars(args), open(os.path.join(pretrained_model_dir, "params.json"), 'wb'))
    
    num_class = num_classes[datasets[dataset]]
    net = get_model("resnet26", num_class, dataset = "imagenet12")
    
    agent = agent_net.resnet(sum(net.layer_config) * 2)
    
    # freeze the original blocks
    flag = True
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and 'parallel_blocks' not in name:
            if flag is True:
                flag = False
            else:
                m.weight.requires_grad = False
    
    _ = net.cuda()
    _ = agent.cuda()
    
    net_params   = list(filter(lambda p: p.requires_grad, net.parameters()))
    agent_params = list(agent.parameters())
    
    net_opt   = torch.optim.SGD(net_params,   lr= args.lr,       momentum=0.9, weight_decay=weight_decays[dataset])
    agent_opt = torch.optim.SGD(agent_params, lr= args.lr_agent, momentum= 0.9, weight_decay= 0.001)
    
    start_epoch = 0
    for epoch in range(start_epoch, start_epoch+args.nb_epochs):
        adjust_learning_rate_net(net_opt, epoch, args)
        adjust_learning_rate_agent(agent_opt, epoch, args)
        
        st_time = time()
        train_acc, train_loss = train(dataset, epoch, train_loaders[datasets[dataset]], net, agent, net_opt, agent_opt)
        test_acc, test_loss   = test(epoch, val_loaders[datasets[dataset]], net, agent, dataset)
        
        results[0:2, epoch, i] = [train_loss, train_acc]
        results[2:4, epoch, i] = [test_loss, test_acc]
        
    state = {
        'net'  : net,
        'agent': agent,
    }
    
    torch.save(state, os.path.join(pretrained_model_dir, dataset + '.t7'))
    np.save(os.path.join(pretrained_model_dir, 'statistics'), results)

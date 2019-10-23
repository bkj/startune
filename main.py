import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import numpy as np
import json
import collections

import imdbfolder as imdbfolder
from spottune_models import *
import models
import agent_net

from utils import *
from gumbel_softmax import *

# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.determinstic = True

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
_ = np.random.seed(args.seed)
_ = torch.manual_seed(args.seed + 111)
_ = torch.cuda.manual_seed(args.seed + 222)


weight_decays = [
    ("aircraft", 0.0005),
    ("cifar100", 0.0),
    ("daimlerpedcls", 0.0005),
    ("dtd", 0.0),
    ("gtsrb", 0.0),
    ("omniglot", 0.0005),
    ("svhn", 0.0),
    ("ucf101", 0.0005),
    ("vgg-flowers", 0.0001),
    ("imagenet12", 0.0001)]

datasets = [
    ("aircraft", 0),
    ("cifar100", 1),
    ("daimlerpedcls", 2),
    ("dtd", 3),
    ("gtsrb", 4),
    ("omniglot", 5),
    ("svhn", 6),
    ("ucf101", 7),
    ("vgg-flowers", 8)]

datasets = collections.OrderedDict(datasets)
weight_decays = collections.OrderedDict(weight_decays)

with open(args.ckpdir + '/weight_decays.json', 'w') as fp:
    json.dump(weight_decays, fp)

def train(dataset, epoch, train_loader, net, agent, net_optimizer, agent_optimizer):
    
    _ = net.train()
    _ = agent.train()
    
    total_step   = len(train_loader)
    tasks_top1   = AverageMeter()
    tasks_losses = AverageMeter()
    
    for i, (images, labels) in enumerate(train_loader):
        
        images, labels = images.cuda(async=True), labels.cuda(async=True)
        images, labels = Variable(images), Variable(labels)
        
        probs  = agent(images)
        action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
        policy = action[:,:,1]
        
        outputs      = net.forward(images, policy)
        _, predicted = torch.max(outputs.data, 1)
        
        correct = predicted.eq(labels.data).cpu().sum()
        loss   = F.cross_entropy(outputs, labels)
        
        tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))
        tasks_losses.update(loss.item(), labels.size(0))
        
        if i % 50 == 0:
            print(json.dumps({
                "mode"        : "train",
                "epoch"       : epoch,
                "total_epoch" : args.nb_epochs,
                "step"        : i,
                "total_step"  : total_step,
                "loss"        : tasks_losses.avg,
                "acc"         : tasks_top1.val,
                "mean_acc"    : tasks_top1.avg
            }))
            sys.stdout.flush()
        
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
            
       	    probs   = agent(images)
            action  = gumbel_softmax(probs.view(probs.size(0), -1, 2))
            policy  = action[:,:,1]
            
            outputs      = net.forward(images, policy)
            _, predicted = torch.max(outputs.data, 1)
            
            correct = predicted.eq(labels.data).cpu().sum()
            loss    = F.cross_entropy(outputs, labels)
            
            tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))
            tasks_losses.update(loss.item(), labels.size(0))
    
    print(json.dumps({
        "mode"     : "test",
        "epoch"    : epoch,
        "loss"     : tasks_losses.avg,
        "acc"      : tasks_top1.val,
        "mean_acc" : tasks_top1.avg
    }))
    sys.stdout.flush()
    
    return tasks_top1.avg, tasks_losses.avg


def load_weights_to_flatresnet(source, net):
    
    net_old = torch.load(source)['net']
    
    # --
    # Load from disk
    
    store_data = []
    t = 0
    for name, m in net_old.named_modules():
        if isinstance(m, nn.Conv2d):
            store_data.append(m.weight.data)
            t += 1
    
    # --
    # Copy to non-parallel_blocks
    
    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and 'parallel_blocks' not in name:
            m.weight.data = torch.nn.Parameter(store_data[element].clone())
            element += 1
    
    # --
    # Copy to parallel_blocks
    element = 1
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and 'parallel_blocks' in name:
            m.weight.data = torch.nn.Parameter(store_data[element].clone())
            element += 1
    
    # --
    # Copy BatchNorm stats
    
    store_data      = []
    store_data_bias = []
    store_data_rm   = []
    store_data_rv   = []
    for name, m in net_old.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            store_data.append(m.weight.data)
            store_data_bias.append(m.bias.data)
            store_data_rm.append(m.running_mean)
            store_data_rv.append(m.running_var)
    
    # --
    # Copy to non-parallel blocks
    
    element = 0
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d) and 'parallel_block' not in name:
                m.weight.data  = torch.nn.Parameter(store_data[element].clone())
                m.bias.data    = torch.nn.Parameter(store_data_bias[element].clone())
                m.running_var  = store_data_rv[element].clone()
                m.running_mean = store_data_rm[element].clone()
                element += 1
    
    # --
    # Copy to parallel blocks
    
    element = 1
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d) and 'parallel_block' in name:
                m.weight.data  = torch.nn.Parameter(store_data[element].clone())
                m.bias.data    = torch.nn.Parameter(store_data_bias[element].clone())
                m.running_var  = store_data_rv[element].clone()
                m.running_mean = store_data_rm[element].clone()
                element += 1
    
    del net_old
    return net

# --
# Run

train_loaders, valid_loaders, num_classes = imdbfolder.prepare_data_loaders(
    dataset_names=datasets.keys(),
    data_dir=args.datadir,
    imdb_dir=args.imdbdir, 
)

for i, dataset in enumerate(datasets.keys()):
    print("dataset=%s" % dataset)
    
    torch.cuda.manual_seed_all(args.seed)
    
    train_loader = train_loaders[datasets[dataset]]
    valid_loader = valid_loaders[datasets[dataset]]
    num_class    = num_classes[datasets[dataset]]
    
    pretrained_model_dir = args.ckpdir + dataset
    if not os.path.isdir(pretrained_model_dir):
        os.mkdir(pretrained_model_dir)
    
    results = np.zeros((4, args.nb_epochs, len(num_classes)))
    f = pretrained_model_dir + "/params.json"
    with open(f, 'wb') as fh:
        json.dump(vars(args), fh)
    
    net   = load_weights_to_flatresnet('resnet26_pretrained.t7', resnet26(num_class))
    agent = agent_net.resnet(sum(net.layer_config) * 2)
    
    _ = net.cuda()
    _ = agent.cuda()
    
    # !! Skipping first one?
    flag = True
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and 'parallel_blocks' not in name:
            if flag is True:
                flag = False
            else:
                m.weight.requires_grad = False
    
    net_params   = filter(lambda p: p.requires_grad, net.parameters())
    agent_params = agent.parameters()
    
    net_optimizer   = torch.optim.SGD(net_params, lr=args.lr, momentum=0.9, weight_decay=weight_decays[dataset])
    agent_optimizer = torch.optim.SGD(agent_params, lr= args.lr_agent, momentum= 0.9, weight_decay= 0.001)
    
    start_epoch = 0
    for epoch in range(start_epoch, start_epoch+args.nb_epochs):
        adjust_learning_rate_net(net_optimizer, epoch, args)
        adjust_learning_rate_agent(agent_optimizer, epoch, args)
        
        st_time               = time.time()
        
        train_acc, train_loss = train(dataset, epoch, train_loader, net, agent, net_optimizer, agent_optimizer)
        test_acc, test_loss   = test(epoch, valid_loader, net, agent, dataset)
        
        results[0:2,epoch,i] = [train_loss, train_acc]
        results[2:4,epoch,i] = [test_loss, test_acc]
        
        raise Exception()
        
    state = {
        'net'  : net,
        'agent': agent,
    }
    
    torch.save(state, pretrained_model_dir +'/' + dataset + '.t7')
    np.save(pretrained_model_dir + '/statistics', results)

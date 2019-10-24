import os
import time
import json
import random
import argparse
import collections
import numpy as np

import torch
torch.set_default_tensor_type('torch.DoubleTensor')

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import imdbfolder
import imdbfolder2

from spottune_models import resnet26
from models2 import ResNet2, STResNet2
import agent_net

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

args = parser.parse_args()

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
    
    total_step   = len(train_loader)
    tasks_top1   = AverageMeter()
    tasks_losses = AverageMeter()
    
    for i, task_batch in enumerate(train_loader):
        images = task_batch[0] 
        labels = task_batch[1]
        
        images, labels = images.cuda(async=True), labels.cuda(async=True)
        images, labels = Variable(images).double(), Variable(labels)
        
        probs  = agent(images)
        print('probs.sum()', float(probs.sum()))
        action = gumbel_softmax(probs.view(probs.size(0), -1, 2))
        policy = action[:,:,1]
        
        outputs = net.forward(images, policy)
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(labels.data).cpu().sum()
        tasks_top1.update(correct.item()*100 / (labels.size(0)+0.0), labels.size(0))
        
        # Loss
        loss = criterion(outputs, labels)
        tasks_losses.update(loss.item(), labels.size(0))
        
        if i % 50 == 0:
            print ("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc Val: {:.4f}, Acc Avg: {:.4f}"
                .format(epoch+1, args.nb_epochs, i+1, total_step, tasks_losses.val, tasks_top1.val, tasks_top1.avg))
        
        net_optimizer.zero_grad()
        agent_optimizer.zero_grad()
        
        loss.backward()
        
        net_optimizer.step()
        agent_optimizer.step()
        
        print(float(loss))
        if i == 10:
            os._exit(0)
        
    return tasks_top1.avg , tasks_losses.avg


def test(epoch, val_loader, net, agent, dataset):
    net.eval()
    agent.eval()
    
    tasks_top1 = AverageMeter()
    tasks_losses = AverageMeter() 
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
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
            
    print ("Epoch [{}/{}], Loss: {:.4f}, Acc Val: {:.4f}, Acc Avg: {:.4f}"
        .format(epoch+1, args.nb_epochs, tasks_losses.avg, tasks_top1.val, tasks_top1.avg))
    
    return tasks_top1.avg, tasks_losses.avg

def load_weights_to_flatresnet(source, net):
    net_old = torch.load(source)['net']
    
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

criterion = nn.CrossEntropyLoss()

set_seeds(args.seed)

net_0 = load_weights_to_flatresnet('./tmp2.t7', resnet26(num_class))

new = True

if new:
    net = STResNet2(torch.load('tmp2.t7')['net'])
    with torch.no_grad():
        _ = net.linear.bias.set_(net_0.linear.bias.clone())
        _ = net.linear.weight.set_(net_0.linear.weight.clone())
else:
    net = net_0
# >>

set_seeds(args.seed)

agent_0 = agent_net.resnet(24).double()
del agent_0.parallel_blocks
del agent_0.parallel_ds

if new:
    agent = nn.Sequential(
        ResNet2(nblocks=[1, 1, 1]),
        nn.Linear(256, 24)          # !! I think this is the wrong dimensionality
    ).double()
    
    with torch.no_grad():
        for a, b in zip(agent_0.parameters(), agent.parameters()):
            _ = b.set_(a.data.clone())

else:
    agent = agent_0



# <<



_ = net.cuda().double()
_ = net_0.cuda().double()
_ = agent.cuda().double()

# freeze the original blocks
flag = True
for name, m in net_0.named_modules():
    if isinstance(m, nn.Conv2d) and 'parallel_blocks' not in name:
        if flag is True:
            flag = False
        else:
            m.weight.requires_grad = False

net_params   = filter(lambda p: p.requires_grad, net.parameters())
agent_params = agent.parameters()

net_optimizer   = optim.SGD(net_params, lr= args.lr, momentum=0.9, weight_decay=weight_decays[dataset])
agent_optimizer = optim.SGD(agent_params, lr= args.lr_agent, momentum= 0.9, weight_decay= 0.001)

set_seeds(args.seed)
for epoch in range(args.nb_epochs):
    # adjust_learning_rate_net(net_optimizer, epoch, args)
    # adjust_learning_rate_agent(agent_optimizer, epoch, args)
    
    train_acc, train_loss = train(dataset, epoch, train_loader, net, agent, net_optimizer, agent_optimizer)
    test_acc, test_loss   = test(epoch, valid_loader, net, agent, dataset)

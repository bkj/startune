#!/usr/bin/env python

"""
    copy-model.py
"""

import torch
from torch import nn
from simple_models import SimpleResNet

import sys; sys.path.append('dep')

# --
# Helpers

def assign_modules(old_modules, new_modules):
    assert len(old_modules) == len(new_modules)
    
    with torch.no_grad():
        for old, new in zip(old_modules, new_modules):
            for param_name, old_param in old.named_parameters():
                assert hasattr(new, param_name)
                new_param = getattr(new, param_name)
                assert new_param.shape == old_param.shape
                new_param.data.set_(old_param.data.clone())
            
            if type(old) in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
                new.running_mean.data.set_(old.running_mean.clone())
                new.running_var.data.set_(old.running_var.clone())


# --
# Run

source  = 'resnet26_pretrained.t7'
net_old = torch.load(source)['net']
net_new = SimpleResNet()

net_old = net_old.cpu().eval()
net_new = net_new.cpu().eval()

# --
# Copy modules

old_conv = [m for m in net_old.modules() if isinstance(m, nn.Conv2d)]
new_conv = [m for m in net_new.modules() if isinstance(m, nn.Conv2d)]
assign_modules(old_conv, new_conv)

old_bn   = [m for m in net_old.modules() if isinstance(m, nn.BatchNorm2d)]
new_bn   = [m for m in net_new.modules() if isinstance(m, nn.BatchNorm2d)]
assign_modules(old_bn, new_bn)

# --
# Test

# net     = net.eval()
# x = torch.randn(16, 3, 64, 64) / 100

# net = net.cpu().eval()

# class Identity(nn.Module):
#     def forward(self, x): return x

# net.linear = Identity()

# a = net_new(x)
# b = net(x)
# (a == b).all()

torch.save({'net' : net_new}, 'models/SimpleResNet.t7')
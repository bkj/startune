from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

class Identity(nn.Module):
    def forward(self, x): return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act_fn=Identity()):
        super(ConvBN, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        
        self.bn     = nn.BatchNorm2d(out_channels)
        self.act_fn = act_fn
    
    def forward(self, x):
        return self.act_fn(self.bn(self.conv(x)))


class BasicBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, stride, shortcut):
        super(BasicBlock2, self).__init__()
        
        self.conv = nn.Sequential(
            ConvBN(in_channels, out_channels, stride=stride, act_fn=F.relu),
            ConvBN(out_channels, out_channels, stride=1)
        )
        
        self.shortcut = shortcut
        if self.shortcut == 1:
            self.avgpool = nn.AvgPool2d(2)
        
    def forward(self, x):
        out = self.conv(x)
        
        if self.shortcut:
            x = self.avgpool(x)
            x = torch.cat((x, x * 0), dim=1) # ??
        
        return F.relu(out + x)
    
    def __repr__(self):
        return 'BasicBlock2()'


class ResNet2(nn.Module):
    def __init__(self, width=32, nblocks=[4, 4, 4, 4]):
        super(ResNet2, self).__init__()
        
        self.stem = ConvBN(3, width, stride=1)
        
        self.trunk = nn.Sequential(
            self._make_layer(1 * width, 2 * width, nblocks[0]),
            self._make_layer(2 * width, 4 * width, nblocks[1]),
            self._make_layer(4 * width, 8 * width, nblocks[2]),
        )
        
        self.head = nn.Sequential(
            nn.BatchNorm2d(8 * width),
            nn.ReLU(inplace=True),     # Missing from SpotTune for some reason
            nn.AdaptiveAvgPool2d(1),
            Flatten()
        )
    
    def _make_layer(self, in_channels, out_channels, nblocks):
        layers = [
            BasicBlock2(in_channels, out_channels, stride=2, shortcut=True)
        ] + [
            BasicBlock2(out_channels, out_channels, stride=1, shortcut=False) for _ in range(nblocks - 1)
        ]
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.trunk(x)
        x = self.head(x)
        return x


class STResNet2(ResNet2):
    def __init__(self, model, n_class=100, *args, **kwargs):
        super(STResNet2, self).__init__()
        
        self.stem   = model.stem
        self.trunk  = model.trunk
        self.head   = model.head
        self.linear = nn.Linear(256, n_class)
        
        self.ftrunk = deepcopy(model.trunk)
        for m in self.ftrunk.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.requires_grad = False
    
    def forward(self, x, policy=None):
        trunk  = list(self.trunk)
        ftrunk = list(self.ftrunk)
        
        x = self.stem(x)
        
        offset = 0
        if policy is not None:
            for layer, flayer in zip(trunk, ftrunk):
                for block, fblock in zip(layer, flayer):
                    
                    action = policy[:, offset].contiguous()
                    action = action.double().view(-1, 1, 1, 1)
                    
                    x = ((1 - action) * fblock(x)) + (action * block(x))
                    offset += 1
        
        x = self.head(x)
        x = self.linear(x)
        
        return x

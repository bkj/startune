from rsub import *
from matplotlib import pyplot as plt

import os
import sys
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from fastai.basic_train import Learner
from fastai.script import call_parse, Param
from fastai.layers import LabelSmoothingCrossEntropy
from fastai.callbacks import GeneralScheduler, TrainingPhase
from fastai.callback import annealing_cos
from fastai.vision import ImageList, flip_lr, imagenet_stats
from fastai.metrics import accuracy, top_k_accuracy

from torchvision.models import resnet18, resnet50
from fastai.vision.models.xresnet import xresnet34, xresnet50

from startune.models import ResNet

# <<
# Ranger fixes
sys.path.append('/home/bjohnson/software/jhoward/ranger_imagewoof')
from ranger import Ranger
from mxresnet import mxresnet50, mxresnet18
# >>

torch.backends.cudnn.benchmark = True

# --
# Helpers

def fit_with_annealing(learn, num_epoch, lr, ann_start):
    total_batches  = int(num_epoch * len(learn.data.train_dl))
    phase1_batches = int(total_batches * ann_start)
    phase2_batches = int(total_batches * (1 - ann_start))
    
    sched = GeneralScheduler(learn, [
        TrainingPhase(phase1_batches).schedule_hp('lr', lr),
        TrainingPhase(phase2_batches).schedule_hp('lr', lr, anneal=annealing_cos),
    ])
    learn.callbacks.append(sched)
    learn.fit(num_epoch)

@call_parse
def main(
        lr         : Param(type=float)  = 1e-3,
        size       : Param(type=int)    = 72,
        alpha      : Param(type=float)  = 0.99,
        mom        : Param(type=float)  = 0.9,
        eps        : Param(type=float)  = 1e-6,
        epochs     : Param(type=int)    = 5,
        bs         : Param(type=int)    = 256,
        mixup      : Param(type=float)  = 0.,
        opt        : Param(type=str)    = 'adam',
        arch       : Param(type=str)    = 'mxresnet50',
        lr_find    : Param(type=int)    = 0,
        n_classes  : Param(type=int)    = 100,
        ann_start  : Param(type=float)  = 0.72,
        sched_type : Param(type=str)    = 'one_cycle',
        adjust_lr  : Param(type=int)    = 0,
    ):
    
    
    if opt=='adam': 
        opt_func = partial(torch.optim.Adam, betas=(mom,alpha), eps=eps)
    elif opt=='ranger':
        opt_func = partial(Ranger, betas=(mom,alpha), eps=eps)
    
    path    = '/home/bjohnson/projects/startune/data/decathlon-1.0/data/aircraft'
    n_class = 100
    
    data = (
        ImageList
        .from_folder(path)
        .split_by_folder(train='train', valid='val')
        .label_from_folder()
        .transform((
            [],
            []
        ), size=size)
        .databunch(bs=bs, num_workers=8)
        # .presize(size, scale=(0.9, 1))
        # .normalize(imagenet_stats)
    )
    
    if adjust_lr:
        lr *= (bs / 256) # !! Why?  To disentangle batchsize and learning rate?
    
    # if 'mxresnet' not in arch:
    #     model = globals()[arch](num_classes=n_classes, pretrained=False)
    # else:
    #     model = globals()[arch](c_out=n_classes, sa=1, sym=0)
    model = nn.Sequential(
        ResNet(),
        nn.Linear(256, n_class)
    )
    
    learn = Learner(
        data      = data, 
        model     = model,
        wd        = 1e-2,
        opt_func  = opt_func,
        metrics   = [accuracy],
        bn_wd     = False,
        true_wd   = True,
        loss_func = torch.nn.CrossEntropyLoss(),
        # loss_func = LabelSmoothingCrossEntropy()
    )
    
    # if lr_find:
    #     learn.lr_find(wd=1e-2)
    #     learn.recorder.plot()
    #     show_plot()
    
    # if mixup:
    #     learn = learn.mixup(alpha=mixup)
    
    # learn = learn.to_fp16(dynamic=True)
    
    if sched_type == 'one_cycle':
        learn.fit_one_cycle(epochs, lr, div_factor=10, pct_start=0.3)
    else:
        fit_with_annealing(learn, epochs, lr, ann_start)


#!/bin/bash

# baseline.sh

# --
# Fastai from scratch

CUDA_VISIBLE_DEVICES=5 python -m startune.fastai_from_scratch \
    --epochs      120                                \
    --opt         adam                               \
    --bs          128                                \
    --lr          1e-3                               \
    --mom         0.95                               \
    --sched_type  flat_and_anneal                    \
    --ann_start   0.50

# --
# BKJ from scratch

python -m startune.from_scratch \
    --dataset       aircraft    \
    --lr-milestones 80,100      \
    --epochs        120

# --
# 

# Not reproducing scratch results from the paper
# Not reproducing scratch results from residual_adapters
# (> 20 at 60 epochs, > 30 at 90 epochs)
# Switching residual adapters to resnet seems to degrade things quite a bit (though not all the way)
# Do regression testing here

# !! Special weight decay? Special optimizer?
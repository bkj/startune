#!/bin/bash

# run.sh

# --
# Run spottune and straight on a single dataset

mkdir -p {models,results}/exp1/{spottune,straight}

dataset="vgg-flowers"

CUDA_VISIBLE_DEVICES=5 python -m startune.main      \
    --dataset         $dataset                      \
    --outpath         models/exp1/spottune/$dataset \
    --valid-interval  10                            | tee results/exp1/spottune/$dataset.jl

CUDA_VISIBLE_DEVICES=6 python -m startune.main      \
    --dataset         $dataset                      \
    --outpath         models/exp1/straight/$dataset \
    --straight                                      \
    --valid-interval  10                            | tee results/exp1/straight/$dataset.jl

# --
# Q) How often does an image go through the same route?

# --
# Q) What happens if we change the routing temperature during training/testing?


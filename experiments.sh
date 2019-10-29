#!/bin/bash

# run.sh

# --
# Run

mkdir -p {models,results}/exp1

dataset="vgg-flowers"

CUDA_VISIBLE_DEVICES=5 python -m startune.main   \
    --dataset         $dataset                   \
    --outpath         models/exp1/$dataset       \
    --valid-interval  10                         | tee results/exp1/$dataset.jl


CUDA_VISIBLE_DEVICES=6 python -m startune.main      \
    --dataset         $dataset                      \
    --outpath         models/exp1/$dataset.straight \
    --straight                                      \
    --valid-interval  10                            | tee results/exp1/$dataset.straight.jl
#!/bin/bash

# run.sh

# --
# Install

conda create -n spottune3_env python=3.7 pip -y
conda activate spottune3_env

conda install -y -c pytorch pytorch==1.2.0
conda install -y -c pytorch torchvision

pip install tqdm

# # --
# # Download pretrained model

# mkdir -p models
# wget 'https://drive.google.com/uc?export=download&id=1fiFyfb9f3PqVI4q26tp4bP9yNOFXS1KG' \
#     -O models/resnet26

# --
# Download data

./download_data.sh ./data/
wget https://github.com/srebuffi/residual_adapters/blob/master/decathlon_mean_std.pickle?raw=true \
    -O data/decathlon-1.0/decathlon_mean_std.pickle

# --
# Run

mkdir -p results
CUDA_VISIBLE_DEVICES=6 python simple_main.py --dataset aircraft --train-on-valid | tee results/aircraft.jl
CUDA_VISIBLE_DEVICES=6 python simple_main.py --dataset cifar100 --train-on-valid | tee results/cifar100.jl
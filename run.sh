#!/bin/bash

# run.sh

# --
# Install

conda create -n spottune_env python=2.7 pip -y
conda activate spottune_env

conda install -y -c pytorch pytorch==0.4.1
conda install -y -c pytorch torchvision
conda install -y -c conda-forge pycocotools

# --
# Download pretrained model

mkdir -p models
wget 'https://drive.google.com/uc?export=download&id=1fiFyfb9f3PqVI4q26tp4bP9yNOFXS1KG' \
    -O models/resnet26

# --
# Download data

./download_data.sh ./data/
wget https://github.com/srebuffi/residual_adapters/blob/master/decathlon_mean_std.pickle?raw=true \
    -O data/decathlon-1.0/decathlon_mean_std.pickle

# --
# Run

mkdir -p cv
python main.py
#!/bin/bash

# run.sh

# --
# Install

conda create -n spottune3_env python=3.7 pip -y
conda activate spottune3_env

conda install -y -c pytorch pytorch==1.2.0
conda install -y -c pytorch torchvision

pip install tqdm

pip install -e .

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
# Export model

cp startune/dep/{models.py,config_task.py} ./
python scripts/copy-model.py
rm models.py config_task.py

# --
# Run

mkdir -p {models,results}/tov0

for dataset in aircraft cifar100 daimlerpedcls dtd gtsrb omniglot svhn ucf101 vgg-flowers; do
    CUDA_VISIBLE_DEVICES=6 python -m startune.main \
        --dataset $dataset                         \
        --outpath models/tov0/$dataset.pth         \
        --train-on-valid | tee results/tov0/$dataset.jl
done

# --
# Inspect

mkdir -p predictions

python -m startune.predict --dataset aircraft --mode test > predictions/aircraft.test.jl
python -m startune.predict --dataset aircraft --mode valid > predictions/aircraft.valid.jl

cat predictions/aircraft.valid.jl | jq --slurp '.' > results.json

# --
# Regression testing

CUDA_VISIBLE_DEVICES=6 python -m startune.main \
    --dataset aircraft                         \
    --seed 888                                 \
    --outpath models/aircraft.novalid5.pth | tee results/aircraft.novalid5.jl


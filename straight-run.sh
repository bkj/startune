#!/bin/bash

# straight-run.sh

RUN_NAME="tov1.straight"
mkdir -p predictions/$RUN_NAME/{valid,test}

for dataset in aircraft cifar100 daimlerpedcls dtd gtsrb omniglot svhn ucf101 vgg-flowers; do
    CUDA_VISIBLE_DEVICES=5 python -m startune.predict \
        --dataset $dataset                            \
        --mode    test                                \
        --straight                                    \
        --model   models/$RUN_NAME/$dataset.pth > predictions/$RUN_NAME/test/$dataset.jl
done

for dataset in aircraft cifar100 daimlerpedcls dtd gtsrb omniglot ucf101 vgg-flowers; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m startune.predict \
        --dataset $dataset                            \
        --mode    test                                \
        --straight                                    \
        --model   models/$RUN_NAME/$dataset.pth > predictions/$RUN_NAME/test/$dataset.jl
done

ls predictions/$RUN_NAME/test | wc -l
cat predictions/$RUN_NAME/test/*.jl |\
    jq --slurp -rc '.' > predictions/$RUN_NAME/test/results.json
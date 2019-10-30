# startune

(Mostly) from-scratch reimplementation of [SpotTune](https://github.com/gyhui14/spottune)

## Installation + Usage

See `./run.sh` for details

## Results

```
spottune_error_on_decathlon_test_set = {
    "aircraft"      : 0.3378,
    "cifar100"      : 0.2013,
    "daimlerpedcls" : 0.0268,
    "dtd"           : 0.4250,
    "gtsrb"         : 0.0073,
    "omniglot"      : 0.1062,
    "svhn"          : 0.0359,
    "ucf101"        : 0.4803,
    "vgg-flowers"   : 0.1318,
    
    "imagenet12"    : 0.3968, # !! From paper
}

finetune_error_on_decathlon_test_set = {
    "aircraft"      : 0.340,
    "cifar100"      : 0.210,
    "daimlerpedcls" : 0.032,
    "dtd"           : 0.479,
    "gtsrb"         : 0.008,    
    "omniglot"      : 0.108,
    "svhn"          : XXXXX, # !! Too impatient to wait to finish
    "ucf101"        : 0.484,
    "vgg-flowers"   : 0.145,
    
    "imagenet12"    : 0.396, # !! From paper
}
```

`spottune` is _slightly_ better than standard `finetune` on these datasets.

## Improvements

- Data augmentation
- Better learning rate scheduler
- Better optimizer
- Initialization
- fast.ai tricks?
- fix scaling
- ranger
- MISH?
- FP16? Multi-GPU?
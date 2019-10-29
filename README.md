# startune

(Mostly) from-scratch reimplementation of [SpotTune](https://github.com/gyhui14/spottune)

## Installation + Usage

See `./run.sh` for details

## Results

```
error_rates_on_decathlon_test_set = {
    "aircraft"      : 0.3378,
    "cifar100"      : 0.2013,
    "daimlerpedcls" : 0.0268,
    "dtd"           : 0.4250,
    "gtsrb"         : 0.0073,
    "omniglot"      : 0.1062,
    "svhn"          : 0.0359,
    "ucf101"        : 0.4803,
    "vgg-flowers"   : 0.1318,
    
    "imagenet12"    : 1.0000, # Didn't run base model on ImageNet
                              # From paper, error would be 0.3968
}

accs_on_decathlon_test_set = {k:1 - v for k,v in error_rates_on_decathlon_test_set.items()}
accs_on_decathlon_test_set
```

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
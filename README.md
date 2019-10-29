# startune

(Mostly) from-scratch reimplementation of [SpotTune](https://github.com/gyhui14/spottune)

## Installation + Usage

See `./run.sh` for details

## Results

```
error_rates = {
    "aircraft"      : 0.337833783378,
    "cifar100"      : 0.2013,
    "daimlerpedcls" : 0.0268367346939,
    "dtd"           : 0.425,
    "gtsrb"         : 0.00736342042755,
    "omniglot"      : 0.106223043746,
    "svhn"          : 0.0359557467732,
    "ucf101"        : 0.480306634946,
    "vgg-flowers"   : 0.13189136445,
    
    "imagenet12"    : 1.0,              # Didn't run base model on ImageNet
                                        # From paper, error would be 0.3968
}
```
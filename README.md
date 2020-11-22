# Example Code for OCDA-Driving

This repo provides an example code for OCDA-Driving. It implements the AdaptSeg baseline in the paper. This code is heavily borrowed from [AdaptSegNet](https://github.com/wasidennis/AdaptSegNet).

## Requirements
* [PyTorch](https://pytorch.org/) (version >= 0.4.1)
* [mmcv](https://github.com/open-mmlab/mmcv)

## Dataset
* Download the [C-Driving Dataset](https://drive.google.com/drive/folders/1_uNTF8RdvhS_sqVTnYx17hEOQpefmE2r), and put or link it in the root path of this repo.

## Training
This script performs distributed training on multiple GPUs.
```
sh snapshots/vgg16bn_adaptseg/train_dist.sh
```

## Testing
This script performs testing. You may change the 'domain' variable in the script to test on different domains.
```
sh snapshot/vgg16bn_adaptseg/eval_bdd.sh
```
Note that this repo adopts a different implementation of SyncBN with the original paper, thus the performance may not precisely match that in the paper.

## Citation
```
@inproceedings{compounddomainadaptation,
  title={Open Compound Domain Adaptation},
  author={Liu, Ziwei and Miao, Zhongqi and Pan, Xingang and Zhan, Xiaohang and Lin, Dahua and Yu, Stella X. and Gong, Boqing},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```
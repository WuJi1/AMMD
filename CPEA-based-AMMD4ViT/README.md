# Class-Aware Patch Embedding Adaptation for Few-Shot Image Classification
Official PyTorch implementation of the paper **Class-Aware Patch Embedding Adaptation for Few-Shot Image Classification**.

## Requirements
Python 3.8, Pytorch 1.7.0, timm 0.3.2

## Datasets
Please follow [mini-imagenet-tools](https://github.com/yaoyao-liu/mini-imagenet-tools) to obtain the miniImageNet dataset and put it in ./datasets/mini/.

Please follow [tiered-imagenet-tools](https://github.com/yaoyao-liu/tiered-imagenet-tools) to obtain the tieredImageNet dataset and put it in ./datasets/tiered/.

Please follow [download_cifar_fs.sh](https://github.com/mrkshllr/FewTURE/blob/main/datasets/download_cifar_fs.sh) to obtain the CIFAR-FS dataset and put it in ./datasets/cifarfs/.

Please follow [download_fc100.sh](https://github.com/mrkshllr/FewTURE/blob/main/datasets/download_fc100.sh) to obtain the FC100 dataset and put it in ./datasets/fc100/.


## Pretraining 
Please follow https://github.com/mrkshllr/FewTURE/tree/main to pretrain the backbone ViT-small and put it in ./initialization/miniimagenet.

## Training and inference
Please see ./run.sh.

## Quick start :fire:
- Please refer to https://github.com/mrkshllr/FewTURE/tree/main to download the miniImageNet dataset and the checkpoint of the corresponding pretrained ViT-small model.

- Put them in the corresponding folders.

- Run ./run.sh in bash shell.

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/FushengHao/CPEA/blob/main/LICENSE) file.

## Acknowledgement
This repository is built using components of the [FewTURE](https://github.com/mrkshllr/FewTURE) repository for pretraining and the [FEAT](https://github.com/Sha-Lab/FEAT) repository for training and inference.



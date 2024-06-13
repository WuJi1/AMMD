# Attentive maximum mean discrepancy for few-shot image classification

Official PyTorch implementation of the paper **AMMD: Attentive maximum mean discrepancy for few-shot image classification (PR 2024)**

If you use this code or find this work useful for your research, please cite:

```
@article{WU2024110680,
title = {AMMD: Attentive maximum mean discrepancy for few-shot image classification},
journal = {Pattern Recognition},
pages = {110680},
year = {2024},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2024.110680},
url = {https://www.sciencedirect.com/science/article/pii/S003132032400431X},
author = {Ji Wu and Shipeng Wang and Jian Sun},
}
```

## Overview

<img src='README_imgs/overview.gif' width='800'>

## Code Prerequisites

The following packages are required to run the scripts:

- [PyTorch >= version 1.4](https://pytorch.org)

- [tensorboard](https://www.tensorflow.org/tensorboard)

Some comparing methods may require additional packages to run (e.g, OpenCV in DeepEMD and qpth, cvxpy in MetaOptNet).

## Dataset prepare

The dataset should be placed in dir "./data/dataset_name". For example, "./data/miniimagenet" is in the following format:

```
AMMD
├── data
│   ├── miniImagenet
│   │   ├── train
│   │   │   ├──n01532829
│   │   │   ├──────n0153282900000987.png
│   │   ├── val
│   │   │   ├──
│   │   │   ├──────
│   │   ├── test
│   │   │   ├── 
│   │   │   ├──────
```

The miniimagenet and tieredimagenet-DeepEMD dataset can be downloaded from [FRN](https://drive.google.com/drive/folders/1gHt-Ynku6Yc3mz6aKVTppIfNmzML1sNG). The CIFAR-FS and FC100 datasets can be downloaded from [DeepEMD](https://drive.google.com/drive/folders/1sXJgi9pXo8i3Jj1nk08Sxo6x7dAQjf9u?usp=sharing).

## Train and Test

We follow the pretrain method from FewTure for Swin-Tiny and ViT-Small backbone, MCL for ResNet-12 backbone.

Download the pretrained model weights from [Google Drive](https://drive.google.com/drive/folders/1MWRvIDLRhBB9lL0yfLg84Ynq532gR5P6?usp=sharing) and extract it into the `pretrain/` folder.

Moreover, The train/test config and saved checkpoints are saved in the following format:
```
Dataset_Method_NwayKshot_Backbone_Accuracy (e.g., miniImagenet_MEL_katz_N5K1_R12_67.509)
├── predictions.txt (evaluation acc)
├── config.yaml
├── ebest_Nway_Kshot.txt (validation best epoch .txt)
├── ebest_Nway_Kshot.pth (validation best epocg .pth)
```

Download the snapshot files from [Google Drive](https://drive.google.com/drive/folders/1MWRvIDLRhBB9lL0yfLg84Ynq532gR5P6?usp=sharing) and extract it into the `snapshots/` folder.

### Meta-training

For example, AMMD 5-way 1-shot Swin-Tiny miniimagenet GPU 0
```
python experiments/run_evaluator.py \
  --cfg ./snapshots/ResNet-12/MEL_katz/VanillaFCN/miniImagenet_MEL_katz_N5K1_R12_67.509/MEL_katz_N5K1_R12.yaml \
  -c ./snapshots/ResNet-12/MEL_katz/VanillaFCN/miniImagenet_MEL_katz_N5K1_R12_67.509/ebest_5way_1shot.pth \
  --device 0
```

### Evaluating AMMD

For example, AMMD 5-way 1-shot Swin-Tiny miniimagenet GPU 0
```
python experiments/run_evaluator.py \
  --cfg ./snapshots/ResNet-12/MEL_katz/VanillaFCN/miniImagenet_MEL_katz_N5K1_R12_67.509/MEL_katz_N5K1_R12.yaml \
  -c ./snapshots/ResNet-12/MEL_katz/VanillaFCN/miniImagenet_MEL_katz_N5K1_R12_67.509/ebest_5way_1shot.pth \
  --device 0
```


## Few-shot Classification Results

Experimental results on few-shot learning datasets with ResNet-12/ViT-Small/Swin-Tiny backbone. We report average results with 1,000 randomly sampled episodes for both 1-shot and 5-shot evaluation.

<img src='README_imgs/MCL-basic-compare.png' width='600'>

## Acknowledgement

We thank the following repos providing helpful components/functions in our work.

- [MCL](https://github.com/cyvius96/prototypical-network-pytorch)

- [FewTURE](https://github.com/mrkshllr/FewTURE)

- [CPEA](https://github.com/FushengHao/CPEA)


## Contact

We have tried our best to upload the correct snapshots on the google drive.

If you encounter any issues or have questions about using the code, feel free to contact me [wuji98@stu.xjtu.edu.cn](wuji98@stu.xjtu.edu.cn)

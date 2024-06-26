# Attentive maximum mean discrepancy for few-shot image classification

Official PyTorch implementation of our PR 2024 paper [**AMMD: Attentive maximum mean discrepancy for few-shot image classification**](https://www.sciencedirect.com/science/article/pii/S003132032400431X)

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

<img src='README_imgs/overview.png' width='800'>

## Code Prerequisites

The following packages are required to run the scripts:

- [PyTorch >= version 1.4](https://pytorch.org)

- [tensorboard](https://www.tensorflow.org/tensorboard)

Some comparing methods may require additional packages to run (e.g, OpenCV in DeepEMD and qpth, cvxpy in MetaOptNet).

## Dataset prepare

The dataset should be placed in dir "./data/dataset_name" with the same format. 

For example,  miniimagenet dataset is in the following format:

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

We follow the pretrain method from [FewTURE](https://github.com/mrkshllr/FewTURE) for Swin-Tiny and ViT-Small backbone, [MCL](https://github.com/cyvius96/prototypical-network-pytorch) for ResNet-12 backbone.

Download the pretrain weights from [Google Drive](https://drive.google.com/drive/folders/1Y2mEmOQHcTcKprVlZbtvRgXsPXT7IiD2?usp=drive_link) and extract it into the `pretrain/` folder.

```
Dataset_Method_NwayKshot_Backbone_Accuracy (e.g., miniImagenet_AMMD_linear_triplet_N5K1_R12)
├──Time_Accuracy (e.g., 2023-09-15-21_36_70.319)
├──── predictions.txt (evaluation acc)
├──── config.yaml
├──── ebest_Nway_Kshot.txt (validation best epoch .txt)
├──── ebest_Nway_Kshot.pth (validation best epocg .pth)
```

Moreover, The train/test config and saved checkpoints are saved in the following format as above.

Download the meta-train snapshot from [Google Drive](https://drive.google.com/drive/folders/1CGkmW7rayh5sFjwjgE2w8t4XOValfbLi?usp=drive_link) and extract it into the `snapshots/` folder.

### Meta-training

For example, Swin-Tiny AMMD 5-way 5-shot miniimagenet GPU 0
```
python experiments/run_trainer.py \
  --cfg ./configs_AMMD/miniImagenet/AMMD_linear_triplet_N5K5_swin_0.3_0.5.yaml \
  -pt ./pretrain/Swin/mini \
  --device 0
```

ResNet-12 AMMD 5-way 5-shot miniimagenet GPU 0,1
```
python experiments/run_trainer_multiGPUs.py \
--cfg ./configs_AMMD/miniImagenet/AMMD_linear_triplet_N5K5_R12_0.3_0.2.yaml \
-pt ./pretrain/ResNet/mini \
--d 0,1
```

ViT-Small AMMD 5-way 5-shot miniimagenet GPU 0
```
cd CPEA-based-AMMD4ViT
python main_mmd_train.py --gpu 0 --way 5 --test_way 5 --shot 5 \
  --dataset MiniImageNet \
  --init_weights ../pretrain/ViT/mini/checkpoint1600.pth \
  --loss_gamma 0.1 \
  --temperature 1.0 \
  --max_epoch 100 \
  --exp mmd-mini-main-5way-1shot-0.1-1.0 > mmd-mini-main-5way-1shot-0.1-1.0.txt
```

### Only Evaluating AMMD

For example, ResNet-12 AMMD 5-way 5-shot GPU 0
```
python experiments/run_evaluater.py \
  --cfg ./snapshots/miniImagenet_AMMD_linear_triplet_N5K5_R12/2023-10-27-22_03_85.226/AMMD_linear_triplet_N5K5_R12_0.3_0.2.yaml \
  -c ./snapshots/miniImagenet_AMMD_linear_triplet_N5K5_R12/2023-10-27-22_03_85.226/ebest_5way_5shot.pth \
  -d 0
```

ViT-Small AMMD 5-way 5-shot miniimagenet GPU 0
```
cd CPEA-based-AMMD4ViT
python main_mmd_evaluate.py --gpu 0 --way 5 --test_way 5 --shot 1 \
  --dataset MiniImageNet \
  --init_weights ../pretrain/ViT/mini/checkpoint1600.pth \
  --loss_gamma 0.1 \
  --temperature 1.0 \
  --max_epoch 10 \
  --exp mmd-mini-main-5way-1shot-0.1-1.0 > mmd-mini-main-5way-1shot-0.1-1.0-only_test.txt
```


## Few-shot Classification Results

Experimental results on few-shot learning datasets with ResNet-12/ViT-Small/Swin-Tiny backbone. We report average results with 1,000 randomly sampled episodes for both 1-shot and 5-shot evaluation.

<img src='README_imgs/AMMD_result_1.png' width='600'>

<img src='README_imgs/AMMD_result_2.png' width='600'>

## Acknowledgement

We thank the following repos providing helpful components/functions in our work.

- [MCL](https://github.com/cyvius96/prototypical-network-pytorch)

- [FewTURE](https://github.com/mrkshllr/FewTURE)

- [CPEA](https://github.com/FushengHao/CPEA)


## Contact

We have tried our best to upload the correct snapshots on the google drive.

If you encounter any issues or have questions about using the code, feel free to contact me [wuji98@stu.xjtu.edu.cn](wuji98@stu.xjtu.edu.cn).

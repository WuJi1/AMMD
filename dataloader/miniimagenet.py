import os
import os.path as osp
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import random
from PIL import Image
import csv

from .base_datasets import BaseDataset

class MiniImagenet(BaseDataset):
    def __init__(self, cfg, phase="train"):
        super().__init__(cfg, phase)
        self.transform = self.prepare_transform(cfg, phase)

    def prepare_transform(self, cfg, phase):
        norm = transforms.Normalize(
            np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
            np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
        )
        if cfg.model.encoder == "FourLayer_64F":
            if phase == "train":
                t = [
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize([84, 84]),
                    transforms.ToTensor(),
                    norm
                ]
            else:
                t = [
                    transforms.Resize([84, 84]),
                    transforms.ToTensor(),
                    norm
                ]
        else:
            if phase == "train":
                t = [
                    transforms.Resize([92, 92]),
                    transforms.RandomCrop(84),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    norm
                ]
            else:
                t = [
                    transforms.Resize([92, 92]),
                    transforms.CenterCrop(84),
                    transforms.ToTensor(),
                    norm
                ]
        return transforms.Compose(t)
#from randaugment import RandAugment
class MiniImagenet_swin(BaseDataset):
    def __init__(self, cfg, phase="train"):
        super().__init__(cfg, phase)
        self.transform = self.prepare_transform(cfg, phase)

    def prepare_transform(self, cfg, phase):
        image_size = 224   
        if phase == "train":
            t = ([
                transforms.Resize(256),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])),
            ])
        elif phase == "val" or phase == "test":
            t = ([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
            ])

        return transforms.Compose(t)

class MiniImagenet_vit(BaseDataset):
    def __init__(self, cfg, phase="train"):
        super().__init__(cfg, phase)
        self.transform = self.prepare_transform(cfg, phase)

    def prepare_transform(self, cfg, phase):
        # if cfg.data.img_size == 360:
        #     img_resize = 224
        # else:
        #     img_resize = cfg.data.img_size
        # if phase == "train" or phase == "val" or phase == "test":
        #     t = [
        #         transforms.Resize([cfg.data.img_size, cfg.data.img_size],interpolation=3),
        #         transforms.CenterCrop(img_resize),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet standard
        #     ]
        image_size = 224   
        if phase == "train":
            t = ([
                transforms.Resize(256),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])),
            ])
        elif phase == "val" or phase == "test":
            t = ([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
            ])

        return transforms.Compose(t)
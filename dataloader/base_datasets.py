import os
import os.path as osp
import pickle as p
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import random
from PIL import Image
from abc import ABC, abstractmethod

class BaseDataset(data.Dataset, ABC):
    def __init__(self, cfg, phase="train"):
        super().__init__()
        
        self.data_list = self.prepare_data(cfg, phase)
        self.transform = self.prepare_transform(cfg, phase)

    @abstractmethod
    def prepare_transform(self, cfg, phase):
        pass

    def prepare_data(self, cfg, phase):
        if cfg.data.mode == 'folder':
            return self.prepare_data_list_from_folder(cfg, phase)
        elif cfg.data.mode == 'csv':
            return self.prepare_data_list_from_csv(cfg, phase)
        else:
            raise KeyError('not supported')

    def prepare_data_list_from_folder(self, cfg, phase):
       
        folder = osp.join(cfg.data.image_dir, phase)
        
        class_folders = [osp.join(folder, label) \
            for label in os.listdir(folder) \
            if osp.isdir(osp.join(folder, label)) \
        ]
        random.shuffle(class_folders)
        
        class_img_dict = {
            osp.basename(f): [osp.join(f, img) for img in os.listdir(f) if (".png" in img or ".jpg" in img)] \
            for f in class_folders
        }
        data_list = self.sample_category(class_img_dict, cfg, phase)
        return data_list
    
    def prepare_data_list_from_csv(self, cfg, phase):
        root_dir = cfg.data.image_dir
        csv_path = os.path.join(root_dir, phase + '.csv')
        data_label_info = pd.read_csv(csv_path)
        data_info = list(data_label_info.iloc[:, 0])
        label_info = list(data_label_info.iloc[:, 1])
        class_img_dict = {}
        for img, tar in zip(data_info, label_info):
            if tar not in class_img_dict:
                class_img_dict[tar] = [os.path.join(root_dir, img)]
            else:
                class_img_dict[tar].append(os.path.join(root_dir, img))
        # shuffle
        keys = list(class_img_dict.keys())
        random.shuffle(keys)
        shuffle_class_img_dict = {}
        for k in keys:
            shuffle_class_img_dict[k] = class_img_dict.get(k)
        data_list = self.sample_category(shuffle_class_img_dict, cfg, phase)
        return data_list



    def sample_category(self, class_img_dict, cfg, phase):
        class_list = class_img_dict.keys()

        data_list = []
        query_per_class_per_episode = cfg.train.query_per_class_per_episode if phase == "train" else cfg.test.query_per_class_per_episode
        if phase == "train":
            episode_per_epoch = cfg.train.episode_per_epoch
        elif phase == "val":
            episode_per_epoch = cfg.val.episode
        else:
            episode_per_epoch = cfg.test.episode

        for e in range(episode_per_epoch):
            episode = []
            classes = random.sample(class_list, cfg.n_way)
            for t, c in enumerate(classes):
                imgs_set = class_img_dict[c]
                imgs_select = random.sample(imgs_set, cfg.k_shot + query_per_class_per_episode)
                random.shuffle(imgs_select)
                support_x = imgs_select[:cfg.k_shot]
                query_x = imgs_select[cfg.k_shot:]

                episode.append({
                    "support_x": support_x,
                    "query_x": query_x,
                    "target": t
                })
            data_list.append(episode)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        episode = self.data_list[index]
        support_x, support_y, query_x, query_y = [], [], [], []
        for e in episode:
            query_ = e["query_x"]
            for q in query_:
                im = self.transform(Image.open(q).convert("RGB"))
                query_x.append(im.unsqueeze(0))
            support_ = e["support_x"]
            for s in support_:
                im = self.transform(Image.open(s).convert("RGB"))
                support_x.append(im.unsqueeze(0))
            target = e["target"]
            support_y.extend(np.tile(target, len(support_)))
            query_y.extend(np.tile(target, len(query_)))

        support_x = torch.cat(support_x, 0)
        query_x = torch.cat(query_x, 0)
        support_y = torch.LongTensor(support_y)
        query_y = torch.LongTensor(query_y)

        randperm = torch.randperm(len(query_y))
        query_x = query_x[randperm]
        query_y = query_y[randperm]
        return support_x, support_y, query_x, query_y

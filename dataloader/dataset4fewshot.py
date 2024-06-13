import os
import os.path as osp
import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import random
from PIL import Image
import pandas as pd

from timm.data import create_transform
from timm.data.transforms import str_to_pil_interp


def predataset_vit(cfg, phase="train"):
    return PreDataset(cfg, phase, build_transform=build_transform_vit)
def predataset(cfg, phase="train"):
    return PreDataset(cfg, phase, build_transform=build_transform_cnn)
def metadataset_vit(cfg, phase="train"):
    return MetaDataset(cfg, phase, build_transform=build_transform_vit)
def metadataset(cfg, phase="train"):
    return MetaDataset(cfg, phase, build_transform=build_transform_cnn)

class FewShotDataset(data.Dataset):
    def __init__(self, cfg, phase="train", build_transform=None):
        super().__init__()
        self.phase = phase
        self.img_size = cfg.data.img_size
        self.build_transform = build_transform
        self.data_list = self.prepare_data(cfg, phase)
        self.transform = self.prepare_transform(cfg, phase)
        self.label = [l for (d, l) in self.data_list]
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
        x_list = []
        y_list = []
        for i, cls in enumerate(class_folders):
            imgs = [osp.join(cls, img) for img in os.listdir(cls) if ".png" in img or ".jpg" in img]
            x_list = x_list + imgs
            y_list = y_list + [i] * len(imgs)
        data_list = list(zip(x_list, y_list))
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
        x_list = []
        y_list = []
        for i, cls in enumerate(keys):
            imgs = class_img_dict.get(cls)
            x_list = x_list + imgs
            y_list = y_list + [i] * len(imgs)

        data_list = list(zip(x_list, y_list))
        return data_list

    def __len__(self):
        raise NotImplementedError
    def __getitem__(self, index):
        raise NotImplementedError
 
    def prepare_transform(self, cfg, phase):
        data_folder = osp.basename(osp.abspath(cfg.data.image_dir))
        if data_folder == "FC100" or data_folder == "CIFAR-FS":
            mean = (0.5071, 0.4866, 0.4409)
            std = (0.2009, 0.1984, 0.2023)
        else:
            mean = np.array([x / 255.0 for x in [125.3, 123.0, 113.9]])
            std = np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])

        if phase == "train":
            t = self.build_transform(is_train=True, mean=mean, std=std, img_size=self.img_size)
        else:
            t = self.build_transform(is_train=False, mean=mean, std=std, img_size=self.img_size)
        return t

class PreDataset(FewShotDataset):
    def __init__(self, cfg, phase="train", build_transform=None):
        super().__init__(cfg, phase, build_transform)
        if phase != 'train':
            self.eposides = get_eposides(self.label, cfg.pre.val_episode, cfg.n_way, cfg.k_shot+cfg.test.query_per_class_per_episode)
    def __len__(self):
        if self.phase == 'train':
            return len(self.data_list)
        else:
            return len(self.eposides)

    def __getitem__(self, index):
        if self.phase == 'train':
            x, y = self.data_list[index]
            im_x = self.transform(Image.open(x).convert("RGB"))
            return im_x, y
        else:
            # assert batch size = 1 for single GPU
            index = self.eposides[index]
            im_x = []
            target = []
            for ind in index:
                x, y = self.data_list[ind]
                img = self.transform(Image.open(x).convert("RGB"))
                im_x.append(img.unsqueeze(0))
                target.append(y)
            im_x = torch.cat(im_x, 0)
            target = torch.LongTensor(target)
            return im_x, target

class MetaDataset(FewShotDataset):
    def __init__(self, cfg, phase="train", build_transform=None):
        super().__init__(cfg, phase, build_transform)
        if phase == 'train':
            eposide = cfg.train.episode_per_epoch
            query_per_class = cfg.train.query_per_class_per_episode
        elif phase == 'val':
            eposide = cfg.val.episode
            query_per_class = cfg.test.query_per_class_per_episode
        elif phase == 'test':
            eposide = cfg.test.episode
            query_per_class = cfg.test.query_per_class_per_episode
        self.n_way = cfg.n_way
        self.k_shot = cfg.k_shot
        self.query_per_class = query_per_class
        self.eposides = get_eposides(self.label, eposide, cfg.n_way, cfg.k_shot+query_per_class)
    
    def __len__(self):
        return len(self.eposides)        

    def __getitem__(self, index):
        index = self.eposides[index]
        im_x = []
        for ind in index:
            x, y = self.data_list[ind]
            img = self.transform(Image.open(x).convert("RGB"))
            im_x.append(img.unsqueeze(0))
        im_x = torch.cat(im_x, 0)
        target = torch.arange(self.n_way).repeat(self.k_shot+self.query_per_class)
        target = torch.LongTensor(target)
        n_support = self.n_way * self.k_shot
        support_x = im_x[:n_support]
        support_y = target[:n_support]
        query_x = im_x[n_support:]
        query_y = target[n_support:]
        
        _, c, h, w = support_x.size()
        # support_x, support_y = self._reshape(support_x, support_y, c, h, w)
        # query_x, query_y = self._reshape(query_x, query_y, c, h, w)

        pos = torch.randperm(len(query_x))
        query_x = query_x[pos]
        query_y = query_y[pos]

        return support_x, support_y, query_x, query_y
 

def get_eposides(label, n_eposides, n_cls, n_per):
    batches = []
    m_ind = []
    for i in range(max(label) + 1):
        ind = np.argwhere(np.array(label) == i).reshape(-1)
        ind = torch.from_numpy(ind)
        m_ind.append(ind)
    for i_batch in range(n_eposides):
        batch = []
        classes = torch.randperm(len(m_ind))[:n_cls]
        for c in classes:
            l = m_ind[c]
            pos = torch.randperm(len(l))[:n_per]
            batch.append(l[pos])
        batch = torch.stack(batch) # label for each index like: [[0,0,0,....], [1,1,1,...], .....]
        batch = batch.t().reshape(-1) # label for each index like: [0,1,2,..n_way-1,0,1,...n_way_1,.....]
        batches.append(batch)
    return batches

def build_transform_vit(is_train, mean, std, img_size):
    # resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        # swin-like augmentation
        transform = create_transform(
            input_size=img_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            interpolation='bicubic',
            mean=mean,
            std=std
        )
        transform.transforms[0] = transforms.RandomResizedCrop([img_size, img_size])
        return transform
    else:   
        t = [
            transforms.Resize([int(256/224*img_size), int(256/224 * img_size)], interpolation=str_to_pil_interp('bicubic')),
            transforms.CenterCrop([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
        return transforms.Compose(t)

def build_transform_cnn(is_train, mean, std, img_size):
    if is_train:
        t = [
            transforms.RandomResizedCrop([img_size, img_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
            ]
    else:
        t = [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
        if img_size > 32:
            t0 = [
                transforms.Resize([int(256/224*img_size), int(256/224 * img_size)]),
                transforms.CenterCrop([img_size, img_size]), 
            ]
            t = t0 + t
    return transforms.Compose(t)
from pandas import cut
import torch
from .grid_datasets import GridDataset
from .base_datasets import BaseDataset
from .dataset4fewshot import predataset, predataset_vit, metadataset, metadataset_vit
from timm.data import Mixup
from .miniimagenet import MiniImagenet,MiniImagenet_swin,MiniImagenet_vit

def _make_distributed_dataloader(cfg, phase, batch_size, distributed_info, dataset,epoch=0):
    
    shuffle = True if phase == 'train' else False
    sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=distributed_info["num_replicas"], rank=distributed_info["rank"], shuffle=shuffle
        )
    sampler.set_epoch(epoch)
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=8, sampler=sampler, batch_size=batch_size
    )
    return dataloader

def make_dataloader_vit(cfg, phase, batch_size, distributed_info=None, epoch=0, pre_or_meta='meta'):
    if pre_or_meta == 'pre':
        dataset = predataset_vit(cfg, phase)
    else:
        dataset = metadataset_vit(cfg, phase)
    if distributed_info is not None:
        return _make_distributed_dataloader(cfg, phase, batch_size, distributed_info, dataset,epoch)
    else:
        shuffle = True if phase == 'train' else False
        return torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=shuffle)


def _decide_dataset(cfg, phase):
    if cfg.model.forward_encoding.startswith("Grid"):
        return GridDataset(cfg, phase)
    elif cfg.model.encoder.startswith("swin"):
        return MiniImagenet_swin(cfg, phase)
    elif cfg.model.encoder.startswith("vit"):
        return MiniImagenet_vit(cfg, phase)
    else:
        return MiniImagenet(cfg, phase)

def make_dataloader(cfg, phase, batch_size, distributed_info=None, epoch=0, pre_or_meta='meta'):
    if pre_or_meta == 'pre':
        dataset = predataset(cfg, phase)
    else:
        dataset = _decide_dataset(cfg, phase)
    if distributed_info is not None:
        return _make_distributed_dataloader(cfg, phase, batch_size, distributed_info, dataset,epoch)
    else:
        shuffle = True if phase == 'train' else False
        return torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=shuffle)

def make_mixup_fn(cfg):
    mixup_fn = Mixup(
            mixup_alpha=cfg.vit_aug.mixup, cutmix_alpha=cfg.vit_aug.cutmix, cutmix_minmax=cfg.vit_aug.cutmix_minmax, 
            prob=cfg.vit_aug.mixup_prob, switch_prob=cfg.vit_aug.mixup_switch_prob, mode=cfg.vit_aug.mixup_mode,
            label_smoothing=cfg.vit_aug.label_smoothing, num_classes=cfg.pre.pretrain_num_class
        )
    return mixup_fn

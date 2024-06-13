import os
import pprint
import numpy as np
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

def ensure_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    # print(pred.size())
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()
    
def count_acc_mmd(logits, label):
    pred = torch.argmin(logits, dim=-1)
    # print(pred.size())
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def triplet_loss(mmd_dis, query_y, thres):
    # pos_dis = torch.gather(mmd_dis, dim=-1, index=query_y.unsqueeze(-1))
    mask = one_hot(mmd_dis, query_y)
    
    #switch 1
    pos_dis = torch.masked_select(mmd_dis, mask)
    neg_dis = torch.masked_select(mmd_dis, ~mask).view(mask.size(0), -1).min(dim=-1)[0]

    #switch 2
    # pos_dis = torch.masked_select(mmd_dis, mask).view(mask.size(0), -1)
    # neg_dis = torch.masked_select(mmd_dis, ~mask).view(mask.size(0), -1)

    losses = F.relu(pos_dis - neg_dis + thres)
    return losses.mean()

def one_hot(mmd_dis, query_y):
    mask = torch.zeros_like(mmd_dis, dtype=torch.bool)
    return mask.scatter(1, query_y.unsqueeze(-1), 1)
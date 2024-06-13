import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import modules.registry as registry
from modules.utils import batched_index_select, _l2norm
from .innerproduct_similarity import InnerproductSimilarity

# 增大support间的差异性
# conv may be self-attention
# https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space

@registry.Query.register("MMD_cross_linear_triplet")
def mmd_gaussian_ce(channels, cfg):
    return MMD(channels,cfg, loss='triplet', kernel='linear')

@registry.Query.register("MMD_cross_linear_ce")
def mmd_gaussian_ce(channels, cfg):
    return MMD(channels,cfg, loss='crossentropy', kernel='linear')

@registry.Query.register("MMD_cross_gaussian_triplet")
def mmd_gaussian_ce(channels, cfg):
    return MMD(channels, cfg, loss='triplet', kernel='gaussian')

# @registry.Query.register("MMD")
class MMD(nn.Module):

    def __init__(self, in_channels, cfg, loss='crossentropy', kernel='gaussian'):
        super().__init__()

        self.n_way = cfg.n_way
        self.k_shot = cfg.k_shot
        self.kernel = kernel 
        self.l2_norm = cfg.model.mmd.l2norm

        self.loss = loss 
        if self.loss == 'crossentropy':
            self.temperature = cfg.model.temperature
            # self.criterion = nn.CrossEntropyLoss()
        elif self.loss == 'triplet':
            self.thres = cfg.model.tri_thres
        self.cfg = cfg
        self.feat_dim = in_channels
        
        # switch 1: conv
        # self.conv = nn.Conv2d(self.feat_dim, self.feat_dim, 1, bias=False)
        # switch 2: self attention
        self.proj_dim = cfg.model.mmd.proj_dim
        self.key_head = nn.Conv2d(self.feat_dim, self.proj_dim, 1, bias=False)
        self.query_head = nn.Conv2d(self.feat_dim, self.proj_dim, 1, bias=False)
        self.value_head = nn.Conv2d(self.feat_dim, self.feat_dim, 1, bias=False)
        self.head_dim = 32
        self.num_head = self.proj_dim // self.head_dim
        self.scale = (self.proj_dim // self.num_head) ** 0.5

        for l in self.modules():
            if isinstance(l, nn.Conv2d):
                n = l.kernel_size[0] * l.kernel_size[1] * l.out_channels
                torch.nn.init.normal_(l.weight, 0, math.sqrt(2. / n))
                if l.bias is not None:
                    torch.nn.init.constant_(l.bias, 0)
    
    def get_query_key_value(self, x):
        # x: *, c, h, w
        b, n, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        # k: b, n, nH, c, L
        k = self.key_head(x).flatten(-2).view(b, n, self.num_head, self.proj_dim//self.num_head, -1)
        # q: b,n , nH, L, c
        q = self.query_head(x).flatten(-2).view(b, n, self.num_head, self.proj_dim// self.num_head, -1).transpose(-1, -2)
        # v: b,n, nH, L, c
        v = self.value_head(x).flatten(-2).view(b, n, self.num_head, self.feat_dim//self.num_head, -1).transpose(-1, -2)
        return q, k, v

    def cross_attn(self, q, k, v, shortcut, dim):
        h, w = shortcut.shape[-2:]
        b, nq, s = q.shape[:3]
        q = q / self.scale 
        
        attn = F.softmax(q @ k, dim=-1)
        out = attn @ v # b, q, s, nH, L, c
        out = out.transpose(-1,-2).contiguous().view(b, nq, s, -1, h, w)

        shortcut = shortcut.unsqueeze(dim)
        if dim == 1:
            shortcut = shortcut.expand(-1, nq, -1, -1, -1, -1)
        elif dim == 2:
            shortcut = shortcut.expand(-1, -1, s, -1, -1, -1)
        else:
            raise KeyError('the dim is not right')
        out = out + shortcut
        return out

    def forward_attn(self, support, query):
        # support_xf: b, s, c, h, w (b: num of tasks, s:n_way * n_shot, c: channel, h=num of features of each images, w=1)
        # query_xf: b, q, c, h, w (b: num of tasks, q: num_query, c, h, w=1)
        # import ipdb 
        # ipdb.set_trace()
        b, q, c, h, w = query.shape 
        s = support.shape[1]
        #########
        # b, s, nH, L, c or b, s, nH, c, L 
        support_q, support_k, support_v = self.get_query_key_value(support)
        # b, q, s, nH, L, c or b, q, s, nH, c, L 
        support_q, support_k, support_v = map(lambda x: x.unsqueeze(1).expand(-1, q, -1, -1, -1, -1), [support_q, support_k, support_v])
        # b, q, nH, L, c or b, q, nH, c, L 
        query_q, query_k, query_v = self.get_query_key_value(query)
        # b, q, s, nH, L, c or b, q, s, nH, c, L 
        query_q, query_k, query_v = map(lambda x: x.unsqueeze(2).expand(-1, -1, s, -1, -1, -1), [query_q, query_k, query_v])

        #######
        # query guided aggregation of support. of shape [b, q, s, c, h, w]
        support_by_query = self.cross_attn(query_q, support_k, support_v, support, dim=1)
        support_by_query = support_by_query.view(b, q, self.n_way, self.k_shot, c, h, w).permute(0, 1, 2, 3, 5, 6, 4).contiguous().view(b, q, self.n_way, -1, c)
        # support guided aggregation of query, of shape [b, q, s, c, h, w]
        query_by_support = self.cross_attn(support_q, query_k, query_v, query, dim=2)
        query_by_support = query_by_support.view(b, q, self.n_way, self.k_shot, c, h, w).permute(0, 1, 2, 3, 5, 6, 4).contiguous().view(b, q, self.n_way, -1, c)
        return support_by_query, query_by_support


    def forward(self, support_xf, support_y, query_xf, query_y):
        # support_xf: b, s, c, h, w (b: num of tasks, s:n_way * n_shot, c: channel, h=num of features of each images, w=1)
        # query_xf: b, q, c, h, w (b: num of tasks, q: num_query, c, h, w=1)
        # support_xf, query_xf = self.forward_conv(support_xf, query_xf)
        b, q = query_xf.shape[:2]
        # b, q, n_way, kshot*h*w, c
        support_xf, query_xf = self.forward_attn(support_xf, query_xf)
        support_xf, query_xf = centering(support_xf, query_xf)
        if self.l2_norm:
            support_xf = _l2norm(support_xf, dim=-1)
            query_xf = _l2norm(query_xf, dim=-1)
        
        
        mmd_dis = MMD_distance(support_xf, query_xf, self.kernel, self.cfg)

        mmd_dis = mmd_dis.view(b * q, -1) #TODO
        query_y = query_y.view(b * q)
        
        # import ipdb 
        # ipdb.set_trace()
        if self.training:
            if self.loss == 'crossentropy':
                loss = F.cross_entropy(-mmd_dis / self.temperature, query_y)
            elif self.loss == 'triplet':
                # import ipdb 
                # ipdb.set_trace()
                loss = triplet_loss(mmd_dis, query_y, thres=self.thres)
            else:
                raise KeyError('loss function is not supported')
            return {"MMD_loss": loss}
        else:
            _, predict_labels = torch.min(mmd_dis, 1)
            rewards = [1 if predict_labels[j] == query_y[j] else 0 for j in
                       range(len(query_y))]
            return rewards


def MMD_distance(support_xf, query_xf, kernel, cfg):
    assert support_xf.size(-2) == query_xf.size(-2)
    num_fea_per_img = support_xf.size(-2)
    if kernel == 'linear':
        kernel_ss, kernel_qq, kernel_qs = linear_kernel(support_xf, query_xf)
    elif kernel == 'gaussian':
        alphas = [cfg.model.mmd.alphas ** k for k in range(-3, 2)] 
        kernel_ss, kernel_qq, kernel_qs = multi_gaussian_kernel(support_xf, query_xf, alphas=alphas)
    else:
        raise KeyError('kernel is not supported')
    kernel_ss_flatten = kernel_ss.flatten(-2)
    kernel_qq_flatten = kernel_qq.flatten(-2)

    mmd_s = kernel_ss_flatten.sum(dim=-1) - kernel_ss_flatten[:, :, :, ::num_fea_per_img+1].sum(dim=-1)
    mmd_s = 1.0 / (num_fea_per_img*(num_fea_per_img-1)) * mmd_s

    mmd_q = kernel_qq_flatten.sum(dim=-1) - kernel_qq_flatten[:, :, :, ::num_fea_per_img+1].sum(dim=-1)
    mmd_q = 1.0 / (num_fea_per_img*(num_fea_per_img-1)) * mmd_q

    mmd_qs = -2. / (num_fea_per_img * num_fea_per_img) * kernel_qs.flatten(-2).sum(dim=-1)
    
    mmd_dis = mmd_s + mmd_q + mmd_qs
    return mmd_dis

    



def linear_kernel(support_xf, query_xf):
    # https://github.com/jindongwang/transferlearning/blob/master/code/deep/DaNN/mmd.py
    # https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/statistics_diff.py
    # support_xf: b, q, n_way, k_shot*h*w, c
    # query_xf: b, n_query, n_way, k_shot*h*w, c
    # Consider linear time MMD with a linear kernel:
    # K(f(x), f(y)) = f(x)^Tf(y)
    # h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)

    kernel_ss = support_xf @ support_xf.transpose(-1, -2)
    kernel_qq = query_xf @ query_xf.transpose(-1, -2)
    kernel_qs = support_xf @ query_xf.transpose(-1, -2)
    return kernel_ss, kernel_qq, kernel_qs


def multi_gaussian_kernel(support_xf, query_xf, b, n_way, q, c, alphas, n_support_prot, num_fea_per_query):
    distances_ss = torch.cdist(support_xf.view(b * n_way, -1, c), support_xf.view(b*n_way, -1, c)).view(b, n_way, n_support_prot, n_support_prot)
    distances_qq = torch.cdist(query_xf.view(b*q, -1, c), query_xf.view(b*q, -1, c)).view(b, q, num_fea_per_query, num_fea_per_query)
    support_xf_ext = support_xf.unsqueeze(1).expand(-1, q, -1, -1, -1).contiguous()
    query_xf_ext = query_xf.unsqueeze(2).expand(-1, -1, n_way, -1, -1).contiguous()
    distances_qs = torch.cdist(support_xf_ext.view(-1, n_support_prot, c), query_xf_ext.view(-1, num_fea_per_query, c)).view(b, q, n_way, n_support_prot, num_fea_per_query)
    kernels_ss, kernels_qq, kernels_qs = None, None, None
    for alpha in alphas:
        kernels_ss_a, kernels_qq_a, kernels_qs_a = map(lambda x: torch.exp(- alpha * x ** 2), [distances_ss, distances_qq, distances_qs])
        if kernels_ss is None:
            kernels_ss, kernels_qq, kernels_qs = kernels_ss_a, kernels_qq_a, kernels_qs_a
        else:
            kernels_ss = kernels_ss + kernels_ss_a
            kernels_qq = kernels_qq + kernels_qq_a
            kernels_qs = kernels_qs + kernels_qs_a
    return kernels_ss, kernels_qq, kernels_qs


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

def centering(support, query):
    # support: # b, n_way, k_shot*h*w, c
    # query: b, n_query, h*w, c
    support = support - support.mean(dim=-1, keepdim=True)
    query = query - query.mean(dim=-1, keepdim=True)
    return support, query

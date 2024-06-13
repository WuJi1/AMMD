from base64 import encode
from copy import deepcopy
import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import modules.registry as registry
from modules.utils import batched_index_select, _l2norm
from .innerproduct_similarity import InnerproductSimilarity
from timm.models.layers import trunc_normal_
from torch.optim import SGD


# 增大support间的差异性
# https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space

@registry.Query.register("MMD_token_linear_triplet")
def mmd_gaussian_ce(channels, cfg):
    return MMD(channels,cfg, loss='triplet', kernel='linear')

@registry.Query.register("MMD_token_linear_ce")
def mmd_gaussian_ce(channels, cfg):
    return MMD(channels,cfg, loss='crossentropy', kernel='linear')

@registry.Query.register("MMD_token_gaussian_triplet")
def mmd_gaussian_ce(channels, cfg):
    return MMD(channels, cfg, loss='triplet', kernel='gaussian')

class Metaprompt(nn.Module):
    def __init__(self, n_meta_prompt, feat_dim):
        super().__init__()
        self.meta_prompt = nn.Parameter(torch.empty(n_meta_prompt, feat_dim))
        # nn.init.orthogonal_(self.meta_prompt)
        nn.init.kaiming_uniform_(self.meta_prompt, a=math.sqrt(5.0))
        # self.dump_meta_prompt_ = None
    def forward(self):
        return self.meta_prompt

# @registry.Query.register("MMD")
class MMD(nn.Module):

    def __init__(self, in_channels, cfg, loss='crossentropy', kernel='gaussian'):
        super().__init__()

        self.n_way = cfg.n_way
        self.k_shot = cfg.k_shot
        self.kernel = kernel 
        if self.kernel in ['gaussian', 'm_gaussian']:
            self.alphas = [cfg.model.mmd.alphas ** k for k in range(-3, 2)] # same as Learning Transferable Features with Deep Adaptation Networks
        elif self.kernel == 'poly':
            self.weight = cfg.model.mmd.weight# 1.0
            self.pow = cfg.model.mmd.pow # 2.0 
            self.bias = cfg.model.mmd.bias #2.0
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

        self.head_dim = 32
        self.num_head = self.proj_dim // self.head_dim
        self.scale = (self.proj_dim // self.num_head) ** 0.5
        self.num_prompt_per_way = cfg.model.mmd.num_prompt_per_way
        self.n_meta_prompt = self.n_way * self.num_prompt_per_way
        self.meta_prompt = Metaprompt(self.n_meta_prompt, self.feat_dim)# independent from the content of image/ agnostic to class in meta-training set;
        self.dump_meta_prompt = None
        self.transformer = Transformer(dim=self.feat_dim, nhead=self.num_head)
        self.cls_head = nn.Linear(self.feat_dim, self.n_way)
        self.ce = nn.CrossEntropyLoss()
        self.prompt_optim = SGD(
                self.meta_prompt.parameters(), 
                lr=cfg.model.mmd.prompt_lr, 
                momentum=0.9, 
                weight_decay=cfg.model.mmd.prompt_wd,
                nesterov=True
            )


    def updating_prompt(self, support_xf, support_y, query_xf=None, query_y=None):
        # support_xf: b, s, c, h, w (b: num of tasks, s:n_way * n_shot, c: channel, h=num of features of each images, w=1)
        support_xf_ = support_xf.clone().detach()
        if query_xf is not None:
            query_xf_ = query_xf.clone().detach()
            xf = torch.cat([support_xf_, query_xf_], dim=1)
            y = torch.cat([support_y, query_y], dim=1)
        else:
            xf = support_xf_
            y = support_y
        b, s, c, h, w = xf.size()
        if self.training:
            self.dump_meta_prompt = None 
        else:
            if self.dump_meta_prompt is None:
                self.dump_meta_prompt = deepcopy(self.meta_prompt.state_dict())
            # something may go wrong if using DDP
            self.meta_prompt.load_state_dict(self.dump_meta_prompt)

        out_meta_prompt = self.transformer(xf, self.meta_prompt()) # # of shape b, s, num_meta_prompt, c
        
        #switch 1
        out_meta_prompt = out_meta_prompt.mean(dim=-2).view(-1, c)
        logit = self.cls_head(out_meta_prompt)
        cls_loss = self.ce(logit, y.view(b*s)) # may be not supervised
        cls_loss.backward()
        # switch 2 TODO contrastive loss
        #############
        
        self.prompt_optim.step()
        return cls_loss

    def forward_transformer(self, support, query):
        # support_xf: b, s, c, h, w (b: num of tasks, s:n_way * n_shot, c: channel, h=num of features of each images, w=1)
        # query_xf: b, q, c, h, w (b: num of tasks, q: num_query, c, h, w=1)
        # import ipdb 
        # ipdb.set_trace()
        support = self.transformer(support, self.meta_prompt())
        query = self.transformer(query, self.meta_prompt())
        return support, query
    

    def forward(self, support_xf, support_y, query_xf, query_y):
        # support_xf: b, s, c, h, w (b: num of tasks, s:n_way * n_shot, c: channel, h=num of features of each images, w=1)
        # query_xf: b, q, c, h, w (b: num of tasks, q: num_query, c, h, w=1)
        # support_xf, query_xf = self.forward_conv(support_xf, query_xf)
        b, _, c, h, w = support_xf.size()
        _, q, _, _, _ = query_xf.size()
        if self.training:
            with torch.enable_grad():
                cls_loss = self.updating_prompt(support_xf, support_y, query_xf, query_y)
        else:
            with torch.enable_grad():
                cls_loss = self.updating_prompt(support_xf, support_y)
        support_xf, query_xf = self.forward_transformer(support_xf, query_xf) # b, s/q, L, c
        support_xf = support_xf.view(b, self.n_way, -1, c)
        query_xf = query_xf.view(b, q, -1, c)

        n_support_prot = support_xf.size(-2)
        num_fea_per_query = query_xf.size(-2)
        # import ipdb 
        # ipdb.set_trace()
        support_xf, query_xf = centering(support_xf, query_xf)
        if self.l2_norm:
            support_xf = _l2norm(support_xf, dim=-1)
            query_xf = _l2norm(query_xf, dim=-1)
        if self.kernel == 'gaussian':
            kernels_ss, kernels_qq, kernels_qs = multi_gaussian_kernel(support_xf, query_xf, b, self.n_way, q, c, self.alphas, n_support_prot, num_fea_per_query)
        elif self.kernel == 'm_gaussian':
            kernels_ss, kernels_qq, kernels_qs = modified_multi_gaussian_kernel(support_xf, query_xf, b, self.n_way, q, c, self.alphas, n_support_prot, num_fea_per_query)
        elif self.kernel == 'linear':
            kernels_ss, kernels_qq, kernels_qs = linear_kernel(support_xf, query_xf, self.n_way, q)
        elif self.kernel == 'poly':
            kernels_ss, kernels_qq, kernels_qs = poly_kernel(support_xf, query_xf, self.n_way, q, weight=self.weight, pow=self.pow, bias=self.bias)
        kernel_ss_flatten = kernels_ss.flatten(2)
        mmd_s = 1. / (n_support_prot * (n_support_prot - 1)) * (kernel_ss_flatten.sum(dim=-1) - kernel_ss_flatten[:, :, ::n_support_prot+1].sum(dim=-1)) # b, n_way
        kernel_qq_flatten = kernels_qq.flatten(2)
        mmd_q = 1. / (num_fea_per_query * (num_fea_per_query - 1)) * (kernel_qq_flatten.sum(dim=-1) - kernel_qq_flatten[:, :, ::num_fea_per_query+1].sum(dim=-1)) # b, q
        mmd_sq = -2. / (num_fea_per_query * n_support_prot) * kernels_qs.flatten(3).sum(dim=-1) # b, q, n_way
        mmd_dis = mmd_s.unsqueeze(1).expand_as(mmd_sq) + mmd_q.unsqueeze(-1).expand_as(mmd_sq) + mmd_sq
        mmd_dis = mmd_dis.view(b * q, -1) #TODO
        query_y = query_y.view(b * q)
        
        # import ipdb 
        # ipdb.set_trace()
        if self.training:
            #TODO may be not cross entropy
            if self.loss == 'crossentropy':
                loss = F.cross_entropy(-mmd_dis / self.temperature, query_y)
            elif self.loss == 'triplet':
                # import ipdb 
                # ipdb.set_trace()
                loss = triplet_loss(mmd_dis, query_y, thres=self.thres)
            else:
                raise KeyError('Not supported')
            return [{"MMD_loss": loss}, cls_loss]
        else:
            _, predict_labels = torch.min(mmd_dis, 1)
            rewards = [1 if predict_labels[j] == query_y[j] else 0 for j in
                       range(len(query_y))]
            return rewards

class Transformer(nn.Module):
    def __init__(self, dim, nhead, ):
        super().__init__()
        encoder_norm = nn.LayerNorm(dim)
        self.encoder = TransformerEncoder(dim, nhead, encoder_norm)
        decoder_norm = nn.LayerNorm(dim)
        self.decoder = TransformerDecoder(dim, nhead, decoder_norm)
        self.dim = dim 
        self.nhead = nhead
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, query_embed, mask=None, pos_embed=None):
        b, n, c, h, w = src.shape 
        src = src.flatten(-2).view(b * n, c, -1).permute(2, 0, 1) # h*w, b*n, c
        memory = self.encoder(src)
        query_embed = query_embed.unsqueeze(1).repeat(1, b*n, 1)
        out = self.decoder(query_embed, memory)
        out = out.permute(1, 0, 2).contiguous().view(b, n, -1, c) # b,n, num_prompt, c
        return out
        


class TransformerEncoder(nn.Module):
    def __init__(self, dim, nhead, encoder_norm):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nhead)
        self.norm = encoder_norm
    def forward(self, x):
        out = self.attn(query=x, key=x, value=x,)[0]
        out = self.norm(out)
        return out 



class TransformerDecoder(nn.Module):
    def __init__(self, dim, head, decoder_norm):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, head)
        self.norm = decoder_norm
    def forward(self, query, x):
        out = self.attn(query=query, key=x, value=x)[0]
        out = self.norm(out)
        return out




def linear_kernel(support_xf, query_xf, n_way, q):
    # https://github.com/jindongwang/transferlearning/blob/master/code/deep/DaNN/mmd.py
    # support_xf: b, n_way, k_shot*h*w, c
    # query_xf: b, n_query, h*w, c
    # Consider linear time MMD with a linear kernel:
    # K(f(x), f(y)) = f(x)^Tf(y)
    # h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
    kernel_ss = support_xf @ support_xf.transpose(-1, -2)
    kernel_qq = query_xf @ query_xf.transpose(-1, -2)
    support_xf_ext = support_xf.unsqueeze(1).expand(-1, q, -1, -1, -1).contiguous()  # b, q, n_way, k_shot*h*w, c
    query_xf_ext = query_xf.unsqueeze(2).expand(-1, -1, n_way, -1, -1).contiguous()  # b, q, n_way, h*w, c
    kernel_qs = support_xf_ext @ query_xf_ext.transpose(-1, -2)
    return kernel_ss, kernel_qq, kernel_qs

def poly_kernel(support_xf, query_xf, n_way, q, weight=1.0, pow=2, bias=2.0):
    # https://github.com/jindongwang/transferlearning/blob/master/code/deep/DaNN/mmd.py
    # https://en.wikipedia.org/wiki/Polynomial_kernel
    # support_xf: b, n_way, k_shot*h*w, c
    # query_xf: b, n_query, h*w, c
    # K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
    # import ipdb 
    # ipdb.set_trace()
    kernel_ss = support_xf @ support_xf.transpose(-1, -2)
    kernel_qq = query_xf @ query_xf.transpose(-1, -2)
    support_xf_ext = support_xf.unsqueeze(1).expand(-1, q, -1, -1, -1).contiguous()  # b, q, n_way, k_shot*h*w, c
    query_xf_ext = query_xf.unsqueeze(2).expand(-1, -1, n_way, -1, -1).contiguous()  # b, q, n_way, h*w, c
    kernel_qs = support_xf_ext @ query_xf_ext.transpose(-1, -2)
    # may be learned weight, pow, bias
    kernel_ss, kernel_qq, kernel_qs = map(lambda x: (weight * x + bias) ** pow, [kernel_ss, kernel_qq, kernel_qs])
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

def modified_multi_gaussian_kernel(support_xf, query_xf, b, n_way, q, c, alphas, n_support_prot, num_fea_per_query):
    # modified for the norm of feature = 1
    # e^(- alpha *||x - y|| * 2) = e^(2 *alpha * x^T y - 2 * alpha), omit the constant -2; 
    # euqivant to scale the output of multi_gaussian_kernel with factor e^(2*alpha)
    # support_xf: b, n_way, k_shot*h*w, c
    # query_xf: b, n_query, h*w, c

    kernel_ss = support_xf @ support_xf.transpose(-1, -2)
    kernel_qq = query_xf @ query_xf.transpose(-1, -2)
    support_xf_ext = support_xf.unsqueeze(1).expand(-1, q, -1, -1, -1).contiguous()  # b, q, n_way, k_shot*h*w, c
    query_xf_ext = query_xf.unsqueeze(2).expand(-1, -1, n_way, -1, -1).contiguous()  # b, q, n_way, h*w, c
    kernel_qs = support_xf_ext @ query_xf_ext.transpose(-1, -2)
    
    kernels_ss, kernels_qq, kernels_qs = None, None, None
    for alpha in alphas:
        kernels_ss_a, kernels_qq_a, kernels_qs_a = map(lambda x: torch.exp(2 * alpha * x), [kernel_ss, kernel_qq, kernel_qs])
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

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers.distances.mmd_distance import MMDDistance
from modules.layers.distances.heterommd_distance import HeteroMMDDistance
from modules.layers.attention import Attention,Multi_Cross_Attention
from modules.utils.utils import Metaprompt, _l2norm, centering, triplet_loss, SupConLoss


class MMD(nn.Module):
    def __init__(self, in_channels, cfg, loss="ce", kernel="linear"):
        super().__init__()
        self.n_way = cfg.n_way
        self.k_shot = cfg.k_shot
        self.kernel = kernel
        self.l2_norm = cfg.model.mmd.l2norm
        self.loss = loss
        if self.loss == 'ce':
            self.temperature = cfg.model.temperature
        elif self.loss == 'triplet':
            self.threshold = cfg.model.tri_thres
            self.temperature = cfg.model.temperature
        elif self.loss == 'supcon':
            self.temperature = cfg.model.temperature
        elif self.loss == 'ce_triplet':
            self.temperature = cfg.model.temperature
            self.threshold = cfg.model.tri_thres
        self.cfg = cfg
        self.feat_dim = in_channels
        self.num_head = 8
        self.num_groups = cfg.model.mmd.num_groups
        if cfg.model.mmd.num_groups == 1:
            self.mmd = MMDDistance(cfg, kernel=self.kernel)
        else:
            self.mmd = HeteroMMDDistance(cfg, kernel=self.kernel)
            self.group_dim = self.feat_dim // self.num_groups

    def forward(self, support_xf, support_y, query_xf, query_y):
        raise NotImplementedError

    def compute_loss(self, mmd_dis, query_y):
        if self.loss == 'ce':
            loss = F.cross_entropy(-mmd_dis / self.temperature, query_y)
        elif self.loss == 'NLL':
            loss_f = nn.NLLLoss()
            loss = loss_f(mmd_dis/0.5, query_y)
        elif self.loss == 'triplet':
            loss = triplet_loss(mmd_dis, query_y, thres=self.threshold)
        elif self.loss == 'supcon':
            loss_f = SupConLoss()
            loss = loss_f(mmd_dis, query_y)
        elif self.loss == 'ce_triplet':
            loss = 0.5 * F.cross_entropy(-mmd_dis / self.temperature, query_y) + triplet_loss(mmd_dis, query_y, thres=self.threshold)
        else:
            raise KeyError("loss is not supported")
        return loss


    def inference(self, support_xf, query_xf, query_y, beta=None, gamma=None):
        ns = support_xf.size(1)
        b, nq, nf, c = query_xf.size()
        if self.num_groups > 1:
            support_xf = support_xf.view(b, ns, nf, self.num_groups, -1).permute(0, 1, 3, 2,
                                                                                 4).contiguous()  # b, ns, ng, nf,c
            query_xf = query_xf.view(b, nq, nf, self.num_groups, -1).permute(0, 1, 3, 2, 4).contiguous()
        support_xf, query_xf = centering(support_xf, query_xf)
        if self.l2_norm:
            support_xf = _l2norm(support_xf, dim=-1)
            query_xf = _l2norm(query_xf, dim=-1)

        # switch 1
        if self.cfg.model.mmd.switch == "all_supports":
            if self.num_groups > 1:
                support_xf = support_xf.view(b, self.n_way, -1, self.num_groups, nf,
                                             self.group_dim)  # b, n_way, k_shot*h*w, c_
                support_xf = support_xf.permute(0, 1, 3, 2, 4, 5).reshape(b, self.n_way, self.num_groups, -1,
                                                                          self.group_dim)
            else:
                support_xf = support_xf.reshape(b, self.n_way, -1, c)  # b, n_way, k_shot*h*w, c
            mmd_dis = self.mmd(support_xf, query_xf, beta=beta, gamma=gamma).view(b * nq, -1)

        # switch 2
        elif self.cfg.model.mmd.switch == "per_shot":
            mmd_dis = self.mmd(support_xf, query_xf, beta, gamma)  # b, nq, n_way*kshot
            mmd_dis = mmd_dis.view(b * nq, self.n_way, self.k_shot).mean(-1)  #.median(dim=-1)[0]  # TODO: mean, min, or max
        else:
            raise KeyError("this switch is not supported")
        query_y = query_y.view(b * nq)
        if self.training:
            loss = self.compute_loss(mmd_dis/self.temperature, query_y)
            return {"MMD_loss": loss}
        else:
            _, predict_labels = torch.min(mmd_dis, 1)
            rewards = [1 if predict_labels[j] == query_y[j] else 0 for j in
                       range(len(query_y))]
            return rewards

class MMD_ori(MMD):
    def __init__(self, in_channels, cfg, loss="ce", kernel="linear"):
        super().__init__(in_channels, cfg, loss, kernel)
        self.n_way = cfg.n_way
        self.k_shot = cfg.k_shot
        self.kernel = kernel
        self.l2_norm = cfg.model.mmd.l2norm
        self.loss = loss
        if self.loss == 'ce':
            self.temperature = cfg.model.temperature
        elif self.loss == 'triplet':
            self.threshold = cfg.model.tri_thres
        elif self.loss == 'supcon':
            self.temperature = cfg.model.temperature
        elif self.loss == 'ce_triplet':
            self.temperature = cfg.model.temperature
            self.threshold = cfg.model.tri_thres
        self.cfg = cfg
        self.feat_dim = in_channels
        self.num_head = 8
        self.num_groups = cfg.model.mmd.num_groups
        if cfg.model.mmd.num_groups == 1:
            self.mmd = MMDDistance(cfg, kernel=self.kernel)
        else:
            self.mmd = HeteroMMDDistance(cfg, kernel=self.kernel)
            self.group_dim = self.feat_dim // self.num_groups

    def forward(self, support_xf, support_y, query_xf, query_y):
        # support_xf: b, num_supp, c, h, w
        # query_xf: b, num_query, c, h, w
        # support_y: b, num_supp
        # query_y: b, num_query
        # support_xf = self.attention(support_xf)  # b, ns, h*w, c
        # query_xf = self.attention(query_xf)  # b, nq, h*w, c
        support_xf = support_xf.permute(0,1,3,4,2)
        support_xf = support_xf.view(support_xf.size(0),support_xf.size(1),-1,support_xf.size(-1))
        query_xf = query_xf.permute(0,1,3,4,2)
        query_xf = query_xf.view(query_xf.size(0),query_xf.size(1),-1,query_xf.size(-1))

        if self.training:
            return self.inference(support_xf, query_xf, query_y)
        else:
            return self.inference(support_xf, query_xf, query_y)

    def compute_loss(self, mmd_dis, query_y):
        if self.loss == 'ce':
            loss = F.cross_entropy(-mmd_dis / self.temperature, query_y)
        elif self.loss == 'NLL':
            loss_f = nn.NLLLoss()
            loss = loss_f(mmd_dis/0.5, query_y)
        elif self.loss == 'triplet':
            loss = triplet_loss(mmd_dis, query_y, thres=self.threshold)
        elif self.loss == 'supcon':
            loss = SupConLoss(mmd_dis, query_y)
        else:
            raise KeyError("loss is not supported")
        return loss


class AttentiveMMDPrompt(MMD):
    def __init__(self, in_channels, cfg, loss="ce", kernel="linear"):
        super().__init__(in_channels, cfg, loss, kernel)
        self.q = nn.Linear(self.feat_dim, self.feat_dim, bias=False)
        self.k = nn.Linear(self.feat_dim, self.feat_dim, bias=False)
        self.num_head = cfg.model.mmd.num_head
        self.attention = Attention(self.feat_dim, self.n_way, self.k_shot, self.num_head,
                                   is_proj=cfg.model.mmd.has_proj)
        # self.alpha = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.alpha.data.fill_(0.4)
        self.alpha = cfg.model.mmd.attention_temperature
        self.att_type = cfg.model.mmd.att_type

        if self.num_groups == 1:
            self.scale = math.sqrt(self.feat_dim)
        else:
            if cfg.model.mmd.scale == 0.0:
                self.scale = math.sqrt(self.group_dim)
            else:
                self.scale = cfg.model.mmd.scale

    def compute_attn_weight(self, global_f, local_f, dim):
        # global_f: b, nq/ns, c
        # # local_f: b, ns/nq, nf, c
        # local_v = local_f
        global_f = self.q(global_f)
        local_f = self.k(local_f)
        global_f, local_f = centering(global_f, local_f)
        global_f = _l2norm(global_f, dim=-1)
        local_f = _l2norm(local_f, dim=-1)

        ng = global_f.size(dim=1)
        nl = local_f.size(dim=1)

        if dim == 1:
            # global_f ==> support
            global_f = global_f.unsqueeze(dim).expand(-1, nl, -1, -1).unsqueeze(-1) # b, nq, ns, c,1
            local_f = local_f.unsqueeze(dim+1).expand(-1, -1, ng, -1, -1) # b, nq, ns, nf,c

        elif dim == 2:
            # global_f ==> query
            global_f = global_f.unsqueeze(dim).expand(-1, -1, nl, -1).unsqueeze(-1) # b, nq, ns, c, 1
            local_f = local_f.unsqueeze(dim-1).expand(-1, ng, -1, -1, -1) # b, nq, ns, nf,c


        if self.num_groups > 1:
            b, nq, ns, nf, c = local_f.size()
            global_f = global_f.view(b, nq, ns, self.num_groups, -1, 1) # b, nq, ns, ng, c_, 1
            local_f = local_f.view(b, nq, ns, nf, self.num_groups, -1).permute(0, 1, 2, 4, 3,5) # b, nq, ns, ng, nf, c_,
            attn_weight = local_f @ global_f # b, nq, ns, ng, nf, 1
        else:
            attn_weight = local_f @ global_f # b, nq, ns, nf, 1

        attn_weight = attn_weight / self.alpha
        attn_weight = F.softmax(attn_weight.squeeze(dim=-1), dim=-1).unsqueeze(-1)
        attn = attn_weight 

        return attn

    def compute_self_attn_weight(self, global_f, local_f, local_dim, dim):#
        # global_f: b, nq/ns, c
        # local_f: b, ns/nq, nf, c
        global_f = self.q(global_f)
        local_f = self.k(local_f)
        global_f, local_f = centering(global_f, local_f)
        global_f = _l2norm(global_f, dim=-1)
        local_f = _l2norm(local_f, dim=-1)


        ng = local_dim.size(dim=1)
        nl = local_f.size(dim=1)

        if dim == 1:
            # global_f ==> support
            global_f = global_f.unsqueeze(dim+1).expand(-1, -1, ng, -1).unsqueeze(-1) # b, nq, ns, c,1
            local_f = local_f.unsqueeze(dim+1).expand(-1, -1, ng, -1, -1) # b, nq, ns, nf,c

        elif dim == 2:
            # global_f ==> query
            global_f = global_f.unsqueeze(dim-1).expand(-1, ng, -1, -1).unsqueeze(-1) # b, nq, ns, c, 1
            local_f = local_f.unsqueeze(dim-1).expand(-1, ng, -1, -1, -1) # b, nq, ns, nf,c

        if self.num_groups > 1:
            b, nq, ns, nf, c = local_f.size()
            global_f = global_f.view(b, nq, ns, self.num_groups, -1, 1) # b, nq, ns, ng, c_, 1
            local_f = local_f.view(b, nq, ns, nf, self.num_groups, -1).permute(0, 1, 2, 4, 3,5) # b, nq, ns, ng, nf, c_,
            attn_weight = local_f @ global_f # b, nq, ns, ng, nf, 1
        else:
            attn_weight = local_f @ global_f # b, nq, ns, nf, 1

        attn_weight = attn_weight / (self.alpha) #self.alpha  #self.alpha #0.2 mel 78

        attn_weight = F.softmax(attn_weight.squeeze(dim=-1), dim=-1).unsqueeze(-1)
        # attn_weight = attn_weight.unsqueeze(-1).repeat(1,1,1,1,local_v.size(4))
        attn = attn_weight #* local_v

        return attn

    def compute_beta_gamma(self, support_xf, query_xf):
        support_xf_ori = support_xf
        query_xf_ori = query_xf
        # support_xf, query_xf = centering(support_xf, query_xf)
        # support_xf = _l2norm(support_xf, dim=-1)
        # query_xf = _l2norm(query_xf, dim=-1)
        b, ns, nf, c = support_xf.size()
        nq = query_xf.size(1)

        if self.cfg.model.mmd.switch == "all_supports":
            support_xf = support_xf.view(b, self.n_way, -1, c) # b, nway, k_shot * nf, c
            if self.cfg.model.mmd.pool_type == 0:
                support_xf_cal = support_xf.transpose(-1, -2).unsqueeze(-1)
                support_xf_global = adaptive_pool(support_xf_cal,support_xf_cal)
                
            elif self.cfg.model.mmd.pool_type == 1:
                support_xf_cal = support_xf.transpose(-1, -2).unsqueeze(-1)
                support_xf_global = adaptive_pool_new(support_xf_cal,support_xf_cal)
                
            elif self.cfg.model.mmd.pool_type == 2:
                support_xf_global = support_xf.mean(dim=-2) # b, nway, c
            else:
                print("pool_type error")
        elif self.cfg.model.mmd.switch == "per_shot":
            if self.cfg.model.mmd.pool_type == 0:
                support_xf_ori = support_xf_ori.transpose(-1, -2).unsqueeze(-1)
                support_xf_cal = support_xf.transpose(-1,-2).unsqueeze(-1)
                support_xf_global = adaptive_pool(support_xf_cal,support_xf_ori)
                
            elif self.cfg.model.mmd.pool_type == 1:
                support_xf_ori = support_xf_ori.transpose(-1, -2).unsqueeze(-1)
                support_xf_cal = support_xf.transpose(-1,-2).unsqueeze(-1)
                support_xf_global = adaptive_pool_new(support_xf_cal,support_xf_ori)
                
            elif self.cfg.model.mmd.pool_type == 2:
                support_xf_global = support_xf.mean(dim=-2) # b, ns, c
                
            else:
                print("pool_type error")
        else:
            raise KeyError("this switch is not supported")
        if self.cfg.model.mmd.pool_type == 0:
            query_xf_ori = query_xf_ori.transpose(-1, -2).unsqueeze(-1)
            query_xf_cal = query_xf.transpose(-1, -2).unsqueeze(-1)
            query_xf_global = adaptive_pool(query_xf_cal, query_xf_ori)
            
        elif self.cfg.model.mmd.pool_type == 1:
            query_xf_ori = query_xf_ori.transpose(-1, -2).unsqueeze(-1)
            query_xf_cal = query_xf.transpose(-1, -2).unsqueeze(-1)
            query_xf_global = adaptive_pool_new(query_xf_cal, query_xf_ori)
            

        elif self.cfg.model.mmd.pool_type == 2:
            query_xf_global = query_xf.mean(dim=-2) # b, nq, c

        beta = self.compute_attn_weight(query_xf_global, support_xf, dim=2)
        gamma = self.compute_attn_weight(support_xf_global, query_xf, dim=1)

        return beta, gamma

    def forward(self, support_xf, support_y, query_xf, query_y):
        ns = support_xf.shape[1]
        ########################################################################
        if self.cfg.model.mmd.AMMD_feature == 0:
            support_xf1 = self.attention(support_xf) # b, ns, h*w, c
            query_xf1 = self.attention(query_xf) # b, nq, h*w, c
            support_xf = support_xf.view(support_xf.size(0),support_xf.size(1),support_xf.size(2),-1).permute(0, 1, 3, 2)
            query_xf = query_xf.view(query_xf.size(0),query_xf.size(1),query_xf.size(2),-1).permute(0, 1, 3, 2)

            beta,gamma = self.compute_beta_gamma(support_xf1,query_xf1)
            
        elif self.cfg.model.mmd.AMMD_feature == 1:
            support_xf = self.attention(support_xf) # b, ns, h*w, c
            query_xf = self.attention(query_xf) # b, nq, h*w, c

            beta,gamma = self.compute_beta_gamma(support_xf,query_xf)
            
        elif self.cfg.model.mmd.AMMD_feature == 2:
            support_xf = self.attention(support_xf) # b, ns, h*w, c
            query_xf = self.attention(query_xf) # b, nq, h*w, c
            beta = None
            gamma = None
            
        elif self.cfg.model.mmd.AMMD_feature == 3:
            support_xf = support_xf.view(support_xf.size(0),support_xf.size(1),support_xf.size(2),-1).permute(0, 1, 3, 2)
            query_xf = query_xf.view(query_xf.size(0),query_xf.size(1),query_xf.size(2),-1).permute(0, 1, 3, 2)

            beta,gamma = self.compute_beta_gamma(support_xf,query_xf)

        if self.cfg.model.mmd.ADGM == 's':
            gamma=torch.ones_like(gamma)/gamma.size(-2)
            if self.training:
                return self.inference(support_xf, query_xf, query_y, beta, gamma)
            else:
                return self.inference(support_xf, query_xf, query_y, beta, gamma)
        elif self.cfg.model.mmd.ADGM == 'q':
            beta=torch.ones_like(beta)/beta.size(-2)
            if self.training:
                return self.inference(support_xf, query_xf, query_y, beta, gamma)
            else:
                return self.inference(support_xf, query_xf, query_y, beta, gamma)
        else:
            if self.training:
                return self.inference(support_xf, query_xf, query_y, beta, gamma)
            else:
                return self.inference(support_xf, query_xf, query_y, beta, gamma)

def adaptive_pool(features, attn_from):
    # features and attn_from are paired feature maps, of same size
    assert features.size() == attn_from.size()
    # b, ns, nf, c = features.size()
    B, N, C, H, W = features.size()
    # assert (attn_from >= 0).float().sum() == N * C * H * W
    attention = torch.einsum('bnchw,bnc->bnhw',
                             [attn_from, nn.functional.adaptive_avg_pool2d(attn_from, (1, 1)).view(B, N, C)])
    attention = attention / attention.view(B, N, -1).sum(2).view(B, N, 1, 1).repeat(1, 1, H, W)
    attention = attention.view(B, N, 1, H, W)
    # output size: B, N, C
    return (features * attention).view(B, N, C, -1).sum(3)


def adaptive_pool_new(features, attn_from):
    # features and attn_from are paired feature maps, of same size
    assert features.size() == attn_from.size()
    # b, ns, nf, c = features.size()
    B, N, C, H, W = features.size()
    attention = torch.einsum('bnchw,bnc->bnhw',
                             [attn_from, nn.functional.adaptive_avg_pool2d(attn_from, (1, 1)).view(B, N, C)])
    attention = attention / (C ** 0.5)
    attention = F.softmax(attention, dim=-2)
    attention = attention.view(B, N, 1, H, W)
    # output size: B, N, C
    return (features * attention).view(B, N, C, -1).sum(3)


if __name__ == "__main__":
    A = torch.ones([1, 10, 100,3,3])
    # b = adaptive_pool(A,A)
    # print(b.size())





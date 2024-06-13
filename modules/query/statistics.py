import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import modules.registry as registry
from modules.utils.utils import batched_index_select, _l2norm
from .innerproduct_similarity import InnerproductSimilarity

# TODO:  Attentive Statistics
@registry.Query.register("STA")
class STA(nn.Module):

    def __init__(self, in_channels, cfg):
        super().__init__()

        self.n_way = cfg.n_way
        self.k_shot = cfg.k_shot
        self.nbnn_topk = cfg.model.nbnn_topk
        self.temperature = cfg.model.temperature
        self.criterion = nn.CrossEntropyLoss()
        self.cfg = cfg

        self.project_dim = 160
        self.feat_dim = in_channels
        # self.key_head = nn.Conv2d(self.feat_dim, self.project_dim, 1, bias=False)
        # self.query_head = nn.Conv2d(self.feat_dim, self.project_dim, 1, bias=False)
        # self.value_head = nn.Conv2d(self.feat_dim, self.feat_dim, 1, bias=False)

        # for l in self.modules():
        #     if isinstance(l, nn.Conv2d):
        #         n = l.kernel_size[0] * l.kernel_size[1] * l.out_channels
        #         torch.nn.init.normal_(l.weight, 0, math.sqrt(2. / n))
        #         if l.bias is not None:
        #             torch.nn.init.constant_(l.bias, 0)

   
    def get_stat(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        quantile = torch.tensor([0, 0.25, 0.5, 0.75, 1], device=x.device)
        quan_x = torch.quantile(x, quantile, dim=-1).permute(1, 2, 3, 0)
        stat = torch.cat([mean, var, quan_x], dim=-1)
        return stat


    def forward(self, support_xf, support_y, query_xf, query_y):
        # b: number of tasks,
        device = support_xf.device
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]
        # import ipdb 
        # ipdb.set_trace()
        support_xf = support_xf.view(b, self.n_way, self.k_shot, c, h, w).permute(0,1,3,2,4,5).contiguous().flatten(-3)
        stat_support = self.get_stat(support_xf).unsqueeze(1).expand(-1, q, -1, -1, -1)
        query_xf = query_xf.flatten(-2).unsqueeze(2)
        
        stat_support = _l2norm(stat_support, dim=-2)
        query_xf = _l2norm(query_xf, dim=-2)
        simi_matrix = query_xf.transpose(-1, -2) @ stat_support
        ########################
        # similarity = simi_matrix.topk(1, dim=-1)[0].view(b, q, self.n_way, -1).sum(dim=-1)
        ###########################
        # similarity = simi_matrix.mean(dim=-1).sum(dim=-1)
        ################################
        similarity = simi_matrix.mean(dim=-1).mean(dim=-1)
        similarity = similarity.view(b * q, -1)
        
        query_y = query_y.view(b * q)
        if self.training:
            loss = self.criterion(similarity / self.temperature, query_y)
            return {"ST_loss": loss}
        else:
            _, predict_labels = torch.max(similarity, 1)
            rewards = [1 if predict_labels[j] == query_y[j].to(predict_labels.device) else 0 for j in
                       range(len(query_y))]
            return rewards

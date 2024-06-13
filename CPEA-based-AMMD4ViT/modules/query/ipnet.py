import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry
from modules.utils.utils import l2distance
from modules.utils.utils import Metaprompt, _l2norm, centering, triplet_loss
from modules.layers.distances.mmd_distance import MMDDistance

@registry.Query.register("IPNet")
class ProtoNet(nn.Module):

    def __init__(self, in_channels, cfg):
        super().__init__()

        self.n_way = cfg.n_way
        self.k_shot = cfg.k_shot
        self.criterion = nn.CrossEntropyLoss()
        self.mmd = MMDDistance(cfg, kernel='linear')

        self.temperature = cfg.model.protonet.temperature
        self.ipnet = cfg.model.ipnet

    def _scores(self, support_xf, support_y, query_xf, query_y):
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        #support_xf = support_xf.view(b, self.n_way, self.k_shot, -1, h, w)
        support_xf = support_xf.view((-1,) + support_xf.shape[-3:])
        support_xf = F.adaptive_avg_pool2d(support_xf, 1).view(b, self.n_way, self.k_shot, c)
        if self.k_shot == 1:
            support_proto = support_xf.mean(-2) # [b, self.n_way, c]
        else:
            # 归一化权重，使每个类别的权重和为1
            support_class = support_xf.mean(dim=-2, keepdim=True).expand(-1,-1,self.k_shot,-1)
            
            total_sum = support_xf.sum(dim=-2, keepdim=True)
            sum_without_self = (total_sum.expand(-1,-1,self.k_shot,-1) - support_xf)/(support_xf.size(-2) - 1)
            
            weights = torch.norm(support_class - sum_without_self,dim = -1)            
            row_max_values, _ = torch.max(weights, dim=2, keepdim=True)
            row_min_values, _ = torch.min(weights, dim=2, keepdim=True)
            weights = 1 - (weights-row_min_values)/(self.ipnet*(row_max_values-row_min_values))
            weights_sum = weights.sum(-1,keepdim=True)
            
            normalized_weights = weights / weights_sum
            # 使用归一化的权重对support_xf进行加权求和
            
            support_proto = torch.sum(support_xf * normalized_weights.unsqueeze(-1), dim=2)  # [b, self.n_way, c]

        query_xf = F.adaptive_avg_pool2d(query_xf.view(-1, c, h, w), 1).view(b, q, c)
        # support_proto, query_xf = centering(support_proto, query_xf)
        # support_proto = _l2norm(support_proto, dim=-1)
        # query_xf = _l2norm(query_xf, dim=-1)
        scores = -l2distance(query_xf.transpose(-2, -1).contiguous(), support_proto.transpose(-2, -1).contiguous())
        scores = scores.view(b * q, -1)
        return scores

    def __call__(self, support_xf, support_y, query_xf, query_y):
        scores = self._scores(support_xf, support_y, query_xf, query_y)
        N = scores.shape[0]
        query_y = query_y.view(N)
        if self.training:
            loss = self.criterion(scores / self.temperature, query_y)
            return {"protonet": loss}
        else:
            _, predict_labels = torch.max(scores, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(N)]
            return rewards
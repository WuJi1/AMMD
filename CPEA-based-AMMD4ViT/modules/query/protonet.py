import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry
from modules.utils.utils import l2distance
from modules.utils.utils import Metaprompt, _l2norm, centering, triplet_loss

@registry.Query.register("ProtoNet")
class ProtoNet(nn.Module):

    def __init__(self, in_channels, cfg):
        super().__init__()

        self.n_way = cfg.n_way
        self.k_shot = cfg.k_shot
        self.criterion = nn.CrossEntropyLoss()

        self.temperature = cfg.model.protonet.temperature

    def _scores(self, support_xf, support_y, query_xf, query_y):
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        #support_xf = support_xf.view(b, self.n_way, self.k_shot, -1, h, w)
        support_xf = support_xf.view((-1,) + support_xf.shape[-3:])
        support_xf = F.adaptive_avg_pool2d(support_xf, 1).view(b, self.n_way, self.k_shot, c)
        support_proto = support_xf.mean(-2) # [b, self.n_way, c]

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


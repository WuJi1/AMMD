import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import modules.registry as registry
from modules.utils.utils import batched_index_select, _l2norm
from .innerproduct_similarity import InnerproductSimilarity
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用CUDA
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置一个具体的随机种子
set_seed(42)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
        
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

@registry.Query.register("CPEA")
class CPEA(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        in_channels = 384
        self.fc1 = Mlp(in_features=in_channels, hidden_features=int(in_channels/4), out_features=in_channels)
        self.fc_norm1 = nn.LayerNorm(in_channels)
        self.k_shot = cfg.k_shot
        self.n_way = cfg.n_way
        self.fc2 = Mlp(in_features=196**2,  hidden_features=256, out_features=1)

    #def forward(self, feat_query, feat_shot):
    def forward(self, feat_shot, support_y, feat_query, query_y):
        # query: Q x n x C
        # feat_shot: KS x n x C
        feat_query = feat_query.squeeze(0).squeeze(-1).transpose(1,2)
        feat_shot = feat_shot.squeeze(0).squeeze(-1).transpose(1,2)
        _, n, c = feat_query.size()
        # print(feat_query.size())

        feat_query = self.fc1(torch.mean(feat_query, dim=1, keepdim=True)) + feat_query  # Q x n x C
        feat_shot  = self.fc1(torch.mean(feat_shot, dim=1, keepdim=True)) + feat_shot  # KS x n x C
        feat_query = self.fc_norm1(feat_query)
        feat_shot  = self.fc_norm1(feat_shot)

        query_class = feat_query[:, 0, :].unsqueeze(1)  # Q x 1 x C
        query_image = feat_query[:, 1:, :]  # Q x L x C

        support_class = feat_shot[:, 0, :].unsqueeze(1)  # KS x 1 x C
        support_image = feat_shot[:, 1:, :]  # KS x L x C

        feat_query = query_image + 2.0 * query_class  # Q x L x C
        feat_shot = support_image + 2.0 * support_class  # KS x L x C

        feat_query = F.normalize(feat_query, p=2, dim=2)
        feat_query = feat_query - torch.mean(feat_query, dim=2, keepdim=True)

        feat_shot = feat_shot.contiguous().reshape(self.k_shot, -1, n -1, c)  # K x S x n x C
        feat_shot = feat_shot.mean(dim=0)  # S x n x C
        feat_shot = F.normalize(feat_shot, p=2, dim=2)
        feat_shot = feat_shot - torch.mean(feat_shot, dim=2, keepdim=True)

        # similarity measure
        results = []
        for idx in range(feat_query.size(0)):
            tmp_query = feat_query[idx]  # n x C
            tmp_query = tmp_query.unsqueeze(0)  # 1 x n x C
            out = torch.matmul(feat_shot, tmp_query.transpose(1, 2))  # S x L x L
            out = out.flatten(1)  # S x L*L
            out = self.fc2(out.pow(2))  # S x 1
            out = out.transpose(0, 1)  # 1 x S
            results.append(out)
        scores = torch.cat(results, dim=0)
        
        N = scores.shape[0]
        query_y = query_y.view(N)
        if self.training:
            eps = 0.1
            one_hot = torch.zeros_like(scores).scatter(1, query_y.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (self.n_way - 1)
            log_prb = F.log_softmax(scores, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.mean()
            # loss = self.criterion(scores / self.temperature, query_y)
            return {"protonet": loss}
        else:
            _, predict_labels = torch.max(scores, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(N)]
            return rewards
        # query_y = query_y.view(b * nq)
        # if self.training:
        #     loss = self.compute_loss(mmd_dis/self.temperature, query_y)
        #     return {"MMD_loss": loss}
        # else:
        #     _, predict_labels = torch.min(mmd_dis, 1)
        #     rewards = [1 if predict_labels[j] == query_y[j] else 0 for j in
        #                range(len(query_y))]
        #     return rewards
        
        
        # return results, None # 75 x 5
    
    



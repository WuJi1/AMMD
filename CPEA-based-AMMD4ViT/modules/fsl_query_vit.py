import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import make_encoder
from .query import make_query

class FSLQuery(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.encoder = make_encoder(cfg)
        self.query = make_query(self.encoder.out_channels, cfg)

    def forward(self, support_x, support_y, query_x, query_y):
        b, s, c, h, w = support_x.shape
        q = query_x.shape[1]

        support_xf = self.encoder(support_x.view(-1, c, h, w))
        query_xf = self.encoder(query_x.view(-1, c, h, w))
        fc, fh, fw = support_xf.shape[-3:]
        support_xf = support_xf.view(b, s, fc, fh, fw)
        query_xf = query_xf.view(b, q, fc, fh, fw)
        
        query = self.query(support_xf, support_y, query_xf, query_y)
        if self.training:
            query = sum(query.values())
        return query


def make_fsl(cfg):
    return FSLQuery(cfg)
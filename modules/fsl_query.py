import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import make_encoder
from .query import make_query
import numpy as np
from modules.layers.cpea import CPEA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def make_fsl(cfg):
    return FSLQuery(cfg)

    
class FSLQuery(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.encoder = make_encoder(cfg)
        self.vit_cpea = CPEA()
        self.query = make_query(self.encoder.out_channels, cfg)
        self.forward_encoding = cfg.model.forward_encoding
        self.pyramid_list = self._parse_encoding_params()
        self.k_shot = cfg.k_shot

    def _parse_encoding_params(self):
        idx = self.forward_encoding.find('-')
        if idx < 0:
            return []
        blocks = self.forward_encoding[idx + 1:].split(',')
        blocks = [int(s) for s in blocks]
        return blocks

    def _pyramid_encoding(self, x):
        b, n, c, h , w = x.shape
        x = x.view(-1, c, h, w)
        feature_list = []
        for size_ in self.pyramid_list:
            feature_list.append(F.adaptive_avg_pool2d(x, size_).view(b, n, c, 1, -1))

        if not feature_list:
            out = x.view(b, n, c, 1, -1)
        else:
            out = torch.cat(feature_list, dim=-1)
        return out
    
    def forward_Grid(self, support_x, support_y, query_x, query_y):
        b, s, grids_sc, h, w = support_x.shape
        grids_s = grids_sc // 3
        _, q, grids_qc  = query_x.shape[:3]
        grids_q = grids_qc // 3
        support_xf = F.adaptive_avg_pool2d(self.encoder(support_x.view(-1, 3, h, w)), 1)
        support_xf = support_xf.view(b, s, grids_s, -1).permute(0, 1, 3, 2).unsqueeze(-1)
        query_xf = F.adaptive_avg_pool2d(self.encoder(query_x.view(-1, 3, h, w)), 1)
        query_xf = query_xf.view(b, q, grids_q, -1).permute(0, 1, 3, 2).unsqueeze(-1)

        
        return support_xf, support_y, query_xf, query_y

    def forward_PyramidFCN(self, support_x, support_y, query_x, query_y):
        b, s, c, h, w = support_x.shape
        q = query_x.shape[1]
        support_xf = self.encoder(support_x.view(-1, c, h, w))
        query_xf = self.encoder(query_x.view(-1, c, h, w))
        fc, fh, fw = support_xf.shape[-3:]
        support_xf = support_xf.view(b, s, fc, fh, fw)
        query_xf = query_xf.view(b, q, fc, fh, fw)

        support_xf = self._pyramid_encoding(support_xf)
        query_xf = self._pyramid_encoding(query_xf)
        
        return support_xf, support_y, query_xf, query_y

    def forward_FCN(self, support_x, support_y, query_x, query_y):
        b, s, c, h, w = support_x.shape
        q = query_x.shape[1]
        support_xf = self.encoder(support_x.view(-1, c, h, w))
        query_xf = self.encoder(query_x.view(-1, c, h, w))
        fc, fh, fw = support_xf.shape[-3:]
        support_xf = support_xf.view(b, s, fc, fh, fw)
        query_xf = query_xf.view(b, q, fc, fh, fw)
        return support_xf, support_y, query_xf, query_y

    def forward_FCN_swin(self, support_x, support_y, query_x, query_y):
        b, s, c, h, w = support_x.shape
        q = query_x.shape[1]
        support_xf = self.encoder(support_x.view(-1, c, h, w))
        query_xf = self.encoder(query_x.view(-1, c, h, w))
        if support_xf.size(-2)==197:
            # import ipdb 
            # ipdb.set_trace()
            support_xf_class = support_xf[:,0].unsqueeze(1).repeat(1,196,1)
            query_xf_class = query_xf[:,0].unsqueeze(1).repeat(1,196,1)
            support_xf = support_xf[:,1:]+support_xf_class
            query_xf = query_xf[:,1:]+query_xf_class
        fhw, fc = support_xf.shape[-2:]
        if fhw == 49:
            fh = fw = int(7)
        else:
            fh = int(fhw)
            fw = int(1)
            
        support_xf = support_xf.view(b, s, fh, fw, fc).permute(0, 1, 4, 2, 3)
        query_xf = query_xf.view(b, q, fh, fw, fc).permute(0, 1, 4, 2, 3)
        return support_xf, support_y, query_xf, query_y

    def forward_FCN_vit(self, support_x, support_y, query_x, query_y):
        
        b, s, c, h, w = support_x.shape
        q = query_x.shape[1]
        support_xf = self.encoder(support_x.view(-1, c, h, w))
        query_xf = self.encoder(query_x.view(-1, c, h, w))
        #####
        query_xf,support_xf = self.vit_cpea(query_xf,support_xf,self.k_shot)
        #####
        fhw, fc = support_xf.shape[-2:]
        if fhw == 49:
            fh = fw = int(7)
        else:
            fh = int(fhw)
            fw = int(1)
        support_xf = support_xf.view(b, s, fh, fw, fc).permute(0, 1, 4, 2, 3)
        query_xf = query_xf.view(b, q, fh, fw, fc).permute(0, 1, 4, 2, 3)
        return support_xf, support_y, query_xf, query_y

    def forward_PyramidFCN_swin(self, support_x, support_y, query_x, query_y):
        b, s, c, h, w = support_x.shape
        q = query_x.shape[1]
        support_xf = self.encoder(support_x.view(-1, c, h, w))
        query_xf = self.encoder(query_x.view(-1, c, h, w))
        fhw, fc = support_xf.shape[-2:]
        if fhw == 49:
            fh = fw = int(7)
        else:
            fh = int(fhw)
            fw = int(1)
        support_xf = support_xf.view(b, s, fh, fw, fc).permute(0, 1, 4, 2, 3)
        query_xf = query_xf.view(b, q, fh, fw, fc).permute(0, 1, 4, 2, 3)

        support_xf = self._pyramid_encoding(support_xf)
        query_xf = self._pyramid_encoding(query_xf)

        return support_xf, support_y, query_xf, query_y

    def forward_feature(self, support_x, support_y, query_x, query_y):
        if self.forward_encoding == "FCN":
            support_xf, support_y, query_xf, query_y = self.forward_FCN_swin(support_x, support_y, query_x, query_y)
        elif self.forward_encoding == "FCN_vit":
            support_xf, support_y, query_xf, query_y = self.forward_FCN_vit(support_x, support_y, query_x, query_y)
        elif self.forward_encoding == "FCN_R12":
            support_xf, support_y, query_xf, query_y = self.forward_FCN(support_x, support_y, query_x, query_y)
        elif self.forward_encoding.startswith("Grid"):
            support_xf, support_y, query_xf, query_y = self.forward_Grid(support_x, support_y, query_x, query_y)
        elif self.forward_encoding.startswith("PyramidFCN"):
            support_xf, support_y, query_xf, query_y = self.forward_PyramidFCN(support_x, support_y, query_x, query_y)
        elif self.forward_encoding.startswith("PFCNswin"):
            support_xf, support_y, query_xf, query_y = self.forward_PyramidFCN_swin(support_x, support_y, query_x, query_y)
        else:
            raise NotImplementedError
        return support_xf, support_y, query_xf, query_y
        
    #def forward(self, support_x, support_y, query_x, query_y,episode): #visualization
    def forward(self, support_x, support_y, query_x, query_y):
        support_xf, support_y, query_xf, query_y = self.forward_feature(support_x, support_y, query_x, query_y)
        query = self.query(support_xf, support_y, query_xf, query_y)
        # if episode == 0:

        #         query_xf_pooled = torch.mean(query_xf, dim=(3, 4))  # Assuming the spatial dimensions are the last two dimensions

        #         # Convert the PyTorch tensor to NumPy array
        #         query_xf_np = query_xf_pooled.cpu().numpy()
        #         query_xf_np = np.squeeze(query_xf_np, axis=0)
        #         # Convert labels to numpy
        #         query_y_np = query_y.cpu().numpy()
        #         query_y_np = np.squeeze(query_y_np, axis=0)
        #         # Perform T-SNE
        #         tsne = TSNE(n_components=2, random_state=42, perplexity=20)
        #         query_xf_tsne = tsne.fit_transform(query_xf_np)
        #         # # Plot the T-SNE embeddings with different colors for different classes
        #         # plt.scatter(query_xf_tsne[:, 0], query_xf_tsne[:, 1], c=query_y_np, cmap='viridis')
        #         # plt.title('T-SNE Visualization of Query Features')
        #         # plt.colorbar()
        #         # plt.savefig('tsne_plot.png')
        #         # plt.show()
                
        #         # 获取唯一类别的数量以自定义颜色映射
        #         num_classes = len(np.unique(query_y_np))

        #         # 设置标记属性以提高可见性和美观性
        #         marker_size = 15

        #         colors = ['red', 'orange', 'blue', 'green', 'purple']
        #         for i in range(5):
        #             class_indices = np.where(query_y_np == i)[0]
        #             plt.scatter(query_xf_tsne[class_indices, 0], query_xf_tsne[class_indices, 1], c=colors[i], s=marker_size, label=f'Class {i}')
        #         plt.title('T-SNE Visualization of Query Features')
        #         # plt.colorbar(ticks=range(num_classes), label='Class')
        #         plt.savefig('tsne_plot.png')
        #         plt.show()
                
        if self.training:   
            if not isinstance(query, tuple):
                loss = sum(query.values())
                return loss
            else:
                loss = sum(query[0].values())
                return loss, query[1]
        else:
            return query

# class FSLQueryParallel(FSLQuery):
#     def __init__(self, cfg, num_gpus):
#         super().__init__(cfg)
#         self.num_gpus = num_gpus

#     def forward_Grid(self, support_x, support_y, query_x, query_y):
#         b, s, grids_sc, h, w = support_x.shape
#         grids_s = grids_sc // 3
#         _, q, grids_qc  = query_x.shape[:3]
#         grids_q = grids_qc // 3
        
#         support_xf = F.adaptive_avg_pool2d(self.encoder(support_x.view(-1, 3, h, w)), 1)
#         support_xf = support_xf.view(b, s, grids_s, -1).permute(0, 1, 3, 2).unsqueeze(-1)
#         query_xf = F.adaptive_avg_pool2d(self.encoder(query_x.view(-1, 3, h, w)), 1)
#         query_xf = query_xf.view(b, q, grids_q, -1).permute(0, 1, 3, 2).unsqueeze(-1)
#         return support_xf, support_y, query_xf, query_y

#     def forward_PyramidFCN(self, support_x, support_y, query_x, query_y):
#         b, s, c, h, w = support_x.shape
#         q = query_x.shape[1]
        
#         support_xf = self.encoder(support_x.view(-1, c, h, w))
#         query_xf = self.encoder(query_x.view(-1, c, h, w))
#         fc, fh, fw = support_xf.shape[-3:]
#         support_xf = support_xf.view(b, s, fc, fh, fw)
#         query_xf = query_xf.view(b, q, fc, fh, fw)

#         support_xf = self._pyramid_encoding(support_xf)
#         query_xf = self._pyramid_encoding(query_xf)
               
#         return support_xf, support_y, query_xf, query_y

#     def forward_FCN(self, support_x, support_y, query_x, query_y):
#         b, s, c, h, w = support_x.shape
#         q = query_x.shape[1]
       
#         support_xf = self.encoder(support_x.view(-1, c, h, w))
#         query_xf = self.encoder(query_x.view(-1, c, h, w))
#         fc, fh, fw = support_xf.shape[-3:]
#         support_xf = support_xf.view(b, s, fc, fh, fw)
#         query_xf = query_xf.view(b, q, fc, fh, fw)
                
#         return support_xf, support_y, query_xf, query_y



# def make_fsl_parallel(cfg, num_gpus):
#     return FSLQueryParallel(cfg, num_gpus)
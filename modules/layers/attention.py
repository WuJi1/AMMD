import torch 
import torch.nn as nn 
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, n_way, k_shot, num_head, is_proj=True):
        super().__init__()
        self.dim = dim 
        self.num_head = num_head
        self.is_proj = is_proj
        self.n_way = n_way
        self.k_shot = k_shot
        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=False)
        if self.is_proj:
            self.proj = nn.Linear(self.dim, self.dim)
        self.scale = (self.dim // self.num_head) ** 0.5
    
    def forward(self, x):
        # x: b, n, c, h, w
        # meta_prompt: num_prompt, c
        b, n, c, h, w = x.size()
        x = x.flatten(-2).transpose(-1, -2).contiguous() # b, n, h*w, c
        x = x.view(b * n, -1, c)
        qkv = self.qkv(x).reshape(b * n, -1, 3, self.num_head, c // self.num_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-1, -2)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b * n, -1, c)
        if self.is_proj:
            out = self.proj(out)
        out = x + out # b*n, -1, c
        out = out.view(b, n, -1, c)
        return out

class Multi_Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super(Multi_Cross_Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim, dim, bias=False)
        self.fc_k = nn.Linear(dim, dim, bias=False)
        self.fc_v = nn.Linear(dim, dim, bias=False)
        self.fc = nn.Linear(dim, 32, bias=False)
        self.scale = (self.dim // self.num_heads) ** 0.5

    def forward(self, x, y):
        Q = self.fc_q(x)
        Q = Q/self.scale
        K, V = self.fc_k(y), self.fc_v(y)
        dim_split = self.dim // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, -1), 0)
        K_ = torch.cat(K.split(dim_split, -1), 0)
        V_ = torch.cat(V.split(dim_split, -1), 0)
        A = F.softmax(Q_ @ K_.transpose(-1, -2),dim=-1)
        #O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        y_ = A @ V_
        y_ = torch.cat(y_.split(Q.size(0),0),-1)
        x = x @ y_.transpose(-1,-2)
        # x = self.fc(x)
        x = x/self.scale
        x = F.softmax(x,dim=-2)
        return x


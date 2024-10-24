# This file is part of D2S paper
# "D2S": https://arxiv.org/abs/2307.15250
from torch import nn
import torch.nn.functional as F
import torch
from typing import Tuple
from copy import deepcopy
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base_model import BaseModel


class D2S(BaseModel):
    default_conf = {
        'descriptor_dim': 256,
        'GNN_no_layers': 5,
        'trainable': True,
    }
    required_data = ['points_descriptors']
    def _init(self, conf):
        self.gnn = AttentionalGNN(
            feature_dim=conf.descriptor_dim, no_layers=self.conf.GNN_no_layers)
        self.mapping = MLP([conf.descriptor_dim, 512, 1024, 1024, 512, 4])

    def _forward(self, data):
        descpt = data['points_descriptors']
        out = self.gnn(descpt)
        pred = {}
        pred['points3D'] = self.mapping(out)
        return pred
    
    def loss(self, pred, data):
        raise NotImplementedError

def MLP(channels:list):
    layers = []
    n_chnls = len(channels)
    for i in range(1, n_chnls):
        layers.append(nn.Conv1d(channels[i-1], channels[i], 
                                kernel_size=1, bias=True))
        if i < n_chnls-1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)
    
    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))

class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, no_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(no_layers)])
        self.no_layers = no_layers

    def forward(self, desc: torch.Tensor):
        for i in range(self.no_layers):
            delta = self.layers[i](desc, desc)
            desc = desc + delta
        return desc

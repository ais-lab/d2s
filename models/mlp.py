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
        self.mapping = MLP([conf.descriptor_dim, 512, 1024, 1024, 512, 4])

    def _forward(self, data):
        descpt = data['points_descriptors']
        pred = {}
        pred['points3D'] = self.mapping(descpt)
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
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from n_pair_miner import *

class pnca_loss(torch.nn.Module):
    """Proxy NCA DML"""
    def __init__(self, n_classes, embed_size, alpha = 32, mrg = 0.1):
        super(pnca_loss, self).__init__()
        #self.pars = opt
        self.proxies = torch.nn.Parameter(torch.randn(n_classes, embed_size) / 8)
        self.smoothing_const = smoothing_const
        self.mrg = 0.1
        self.alpha = 32

    def forward(self, image_embed, labels):
        distances = torch.cdist(image_embed, self.proxies) ** 2
        exp_dist = torch.exp(-self.alpha * (    distances - self.mrg))
        classes
        numerators = exp_dist[range(exp_dist.shape[0]), T]

        denom = exp_dist.sum(dim = 1)
        loss = numerators / denom
        loss = loss.sum()
        return loss

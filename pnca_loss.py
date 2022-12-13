import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

class pnca_loss(torch.nn.Module):
    """Proxy NCA DML"""
    def __init__(self, n_classes, embed_size, alpha = 1, mrg = 1):
        super(pnca_loss, self).__init__()
        #self.pars = opt
        self.proxies = torch.nn.Parameter(torch.randn(n_classes, embed_size) )
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, image_embed, labels):
        image_embed = 3*torch.nn.functional.normalize(image_embed, dim=1)
        proxies = 3*torch.nn.functional.normalize(self.proxies, dim=1)
        distances = torch.cdist(image_embed, proxies) ** 2
        exp_dist = torch.exp(-self.alpha * (    distances - self.mrg))
        #breakpoint()
        numerators = exp_dist[range(exp_dist.shape[0]), labels]
        #breakpoint()
        denom = exp_dist.sum(dim = 1)
        loss = torch.log(numerators / denom)
        # included positive proxy also in denominator, as said to improve perf in proxy nca++
        loss = loss.sum()
        return loss

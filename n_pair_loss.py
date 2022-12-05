import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from n_pair_miner import *

class n_pair_loss(torch.nn.Module):
    """Sohn et al N-Pair DML"""
    def __init__(self, l2_reg, batchminer):
        super(n_pair_loss, self).__init__()
        self.pars = opt
        self.l2_reg = l2_reg

    def forward(self, image_embed, labels):
        anc_ind, pos_ind, neg_ind = n_pair_miner(image_embed, labels)
        loss  = 0
        num1 = 0
        for num1 in range(len(anc_ind)):
            anc = image_embed[anc_ind,:]
            pos = image_embed[pos_ind,:]
            neg_set = image_embed[neg_ind,:]

############ WAS GIVING INCORRECT LOSS AS COMPARED TO REFERENCE NEED TO CHECK #############################

            # inner_sum = a_embs[:,None,:].bmm((n_embs - p_embs[:,None,:]).permute(0,2,1))
            # logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
            # loss_ce = cross_entropy(logit, target)
            # l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size
            # num1 = num1+1

        # REFERENCE IMPLEMENTATION FOR DEBUGGING https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch
        for anchor, positive, negative_set in zip(anchors, positives, negatives):
            a_embs, p_embs, n_embs = batch[anchor:anchor+1], batch[positive:positive+1], batch[negative_set]
            inner_sum = a_embs[:,None,:].bmm((n_embs - p_embs[:,None,:]).permute(0,2,1))
            inner_sum = inner_sum.view(inner_sum.shape[0], inner_sum.shape[-1])
            loss  = loss + torch.mean(torch.log(torch.sum(torch.exp(inner_sum), dim=1) + 1))/len(anchors)
            loss  = loss + self.l2_weight*torch.mean(torch.norm(batch, p=2, dim=1))/len(anchors)


        return loss

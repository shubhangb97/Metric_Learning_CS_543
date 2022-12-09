import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# Refer to https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch

def n_pair_miner(labels):
    labels = labels.detach().cpu().numpy()
    anchors, positives, negatives = [],[],[]

    for i in range(len(labels)):
        anchor = i
        pos    = labels==labels[anchor]

        if np.sum(pos)>1:
            anchors.append(anchor)
            avail_positive = np.where(pos)[0]
            avail_positive = avail_positive[avail_positive!=anchor]
            positive       = np.random.choice(avail_positive)
            positives.append(positive)

    ###
    negatives = []
    for anchor,positive in zip(anchors, positives):
        neg_idxs = [i for i in range(len(labels)) if i not in [anchor, positive] and labels[i] != labels[anchor]]
        # neg_idxs = [i for i in range(len(batch)) if i not in [anchor, positive]]
        #negative_set = np.arange(len(batch))[neg_idxs]
        #negatives.append(negative_set)
        negatives.append(np.array(neg_idxs))

    return anchors, positives, negatives

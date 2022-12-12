import torch
import torchvision
import torch.nn as nn
import numpy as np
import tqdm
import os

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets

def computeLoss(A, X, y_mask):
    """
    Compute the loss function given a subsample of images
    Args:
        A (torch.nn.Parameter): the matrix which must be multiplied
            with every input image to get output embedding
        images ((batch_size x nClasses/2 x 3 x h x w) tensor): the list
            of images given by the data loader. Note that A is 2D and
            has dimensions (d x h*w*3)
    Output:
        loss (tensor of size 1): estimated loss given the input
            set of images
    """
    Ax = torch.mm(X, torch.t(A))
    distances = squared_euclidean_distance(Ax)
    distances.diagonal().copy_(np.inf*torch.ones(len(distances)))
    exp = torch.exp(-distances)
    p_ij = exp / exp.sum(dim = 1)
    p_ij_mask = p_ij * y_mask.float()
    p_i = p_ij_mask.sum(dim = 1)
    loss = -torch.log(torch.masked_select(p_i, p_i != 0)).sum()

    #distances.diagonal().copy_(torch.zeros(len(distances)))
    #margin_diff = (1 - distances) * (~y_mask).float()
    #hinge_loss = torch.clamp(margin_diff, min=0).pow(2).sum(1).mean()
    #loss = -p_i.sum()
    #print(loss.item())
    return loss


def squared_euclidean_distance(vec):
    """
    Get the spairwise squared Euclidean distance matrix

    """
    dot = torch.mm(vec, torch.t(vec))
    norm_sq = torch.diag(dot)
    d = norm_sq[None, :] - 2*dot + norm_sq[:, None]
    d = torch.clamp(d, min = 0.0)  # replace negative values with 0
    return d.float()


dataset = datasets.load_iris()


train_acc_vec, test_acc_vec = [], []
# 10 repeats
for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, 
        train_size=0.7, random_state = i)

    X_train = torch.Tensor(X_train).float()
    X_test = torch.Tensor(X_test).float()
    y_train = torch.Tensor(y_train).long()
    y_test = torch.Tensor(y_test).long()

    nEpochs = 100

    #a = torch.randn(4, 4) #full A
    a = torch.randn(2, 4) #rank-2 transformation
    A = nn.Parameter(a)

    optim = torch.optim.Adam([A], lr=1e-04)

    y_mask = y_train[:, None] == y_train[None, :]
    for epoch in range(nEpochs):
            
        optim.zero_grad()

        loss = computeLoss(A, X_train, y_mask)

        loss.backward()

        optim.step()

        #print(f"Epoch {epoch+1}/{nEpochs}: {loss.item():.4f}")

    # Find the train and test accuracies
    A = A.detach().cpu().numpy()

    X_train_embed = X_train @ A.T
    X_test_embed = X_test @ A.T
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train_embed, y_train)
    train_pred = knn.predict(X_train_embed)
    test_pred = knn.predict(X_test_embed)

    train_acc = accuracy_score(train_pred, y_train)
    test_acc = accuracy_score(test_pred, y_test)

    print(f"---Run {i+1}---")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    train_acc_vec.append(train_acc)
    test_acc_vec.append(test_acc)

print("\nOverall")
print(f"Train: {np.mean(train_acc_vec):.3f} +/- {np.std(train_acc_vec):.3f}")
print(f"Test: {np.mean(test_acc_vec):.3f} +/- {np.std(test_acc_vec):.3f}")



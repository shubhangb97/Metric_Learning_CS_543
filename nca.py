import torch
import torchvision
import torch.nn as nn
import numpy as np
import tqdm
from datasets import *
import eval_dataset
from evaluate import *
from torchvision import transforms

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def squared_euclidean_distance(vec):
    """
    Get the squared pairwise Euclidean distance matrix for vec

    """
    dot = torch.mm(vec, torch.t(vec))
    norm_sq = torch.diag(dot)
    d = norm_sq[None, :] - 2*dot + norm_sq[:, None]
    d = torch.clamp(d, min=0.0)  # replace negative values with 0
    return d.float()


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

#datasetName = {
#    "cub":CUBDataset,
#    "sop":SOPDataset,
#    "car":CARDataset
#}["cub"]    # Change the "cub" on this line to select the correct dataset

#fileToSaveA = "Avalue.pickle"

#dataset = datasetName()
#params = {"batch_size" : 10, "num_workers" : 8, "shuffle":True}
#loader = torch.utils.data.DataLoader(dataset, **params)
#h, w = dataset.getOutputShape()

batch_size = 128
latentDims = 512
nEpochs = 100

#train_data = eval_dataset.load(name = "cars",  root = './data/CARS/', mode = 'train', 
#                            transform = eval_dataset.utils.make_transform())
train_data = eval_dataset.load(name = "cub",  root = './data/CUB/', mode = 'train', 
                            transform = eval_dataset.utils.make_transform())
                                                            
test_data = eval_dataset.load(name = "cub",  root = './data/CUB/', mode = 'eval', 
                            transform = eval_dataset.utils.make_transform())
                                                            

train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = True)

dev = "cuda" if torch.cuda.is_available() else "cpu"

h, w = next(iter(train_loader))[0].shape[-2:]

#X_train = np.ndarray(train_data.I).reshape((-1, h*w*3))
#y_train = train_data.ys
#X_test = np.ndarray(test_data.I).reshape((-1, h*w*3))
#y_test = test_data.ys

a = torch.randn(latentDims, h*w*3) * 1e-04 #latent A
A = nn.Parameter(a)

optim = torch.optim.Adam([A], lr=1e-05)

for epoch in range(nEpochs):
    running_loss = 0

    for images, labels in train_loader:
        
        optim.zero_grad()
        
        X = images.reshape((-1, h*w*3))
        
        y_mask = labels[:, None] == labels[None, :] # pairwise boolean class matrix

        loss = computeLoss(A, X, y_mask)

        loss.backward()

        torch.nn.utils.clip_grad_norm_([A], 1e-04)

        optim.step()

        running_loss += loss.item()
        print(loss.item())

    print(f"Epoch {epoch+1}/{nEpochs}: {running_loss * batch_size / len(train_data) :.4f}")

"""
A = A.detach().cpu().numpy()

X_train_embed = X_train @ A.T
X_test_embed = X_test @ A.T
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_embed, y_train)
train_pred = knn.predict(X_train_embed)
test_pred = knn.predict(X_test_embed)

train_acc = accuracy_score(train_pred, y_train)
test_acc = accuracy_score(test_pred, y_test)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
"""

print("")
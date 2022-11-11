import torch
import torchvision
import torch.nn as nn
import numpy as np
import tqdm
from datasets import *
import eval_dataset
from evaluate import *

def computeLoss(A, images):
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
    loss = torch.sum(0.0*torch.matmul(A, images[0, 0].flatten())) # Dummy code, needs to be filled in
    return loss


def squared_euclidean_distance(vec):
    """
    Get the squared pairwise Euclidean distance matrix for vec

    """
    dot = torch.mm(vec, torch.t(vec))
    norm_sq = torch.diag(dot)
    d = norm_sq[None, :] - 2*dot + norm_sq[:, None]
    d = torch.clamp(d, min=0)  # replace negative values with 0
    return d.float()


def computeLossNew(A, X, y):
    """
    Compute the loss sum_i log(pi) # Eq 6 in the paper

    """
    X = X.view(X.shape[0], -1) # flatten channels
    
    y_mask = y[:, None] == y[None, :] # pairwise boolean class matrix

    Ax = torch.mm(X, torch.t(A))
    
    distances = squared_euclidean_distance(Ax)

    distances.diagonal().copy_(np.inf*torch.ones(len(distances)))

    exp = torch.exp(-distances)

    p_ij = exp / exp.sum(dim = 1)

    p_ij_mask = p_ij * y_mask.float()

    p_i = p_ij_mask.sum(dim = 1)

    loss = -torch.log(torch.masked_select(p_i, p_i != 0)).sum()

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

latentDims = 16
nEpochs = 30
trainset = eval_dataset.load(name = "cub",  root = './data/CUB/', mode = 'train', 
                            transform = eval_dataset.utils.make_transform())
loader = torch.utils.data.DataLoader(trainset, batch_size = 32, shuffle = True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
h, w = next(iter(loader))[0].shape[-2:]
A = nn.Parameter(torch.eye(latentDims, h*w*3, requires_grad=True))
optim = torch.optim.Adam([A], lr = 1e-05)
for epoch in range(nEpochs):
    for images, labels in loader:
        optim.zero_grad()
        loss = computeLossNew(A, images, labels)
        loss.backward()
        optim.step()

        print("Loss:", loss.item())

    print(f"Epoch {epoch+1}/{nEpochs}: {loss.item():.4f}")
#torch.save({"A":A}, fileToSaveA)

"""
testset= eval_dataset.load(name = ,  root = './data/', mode = 'eval',
transform = dataset.utils.make_transform( is_train = False, is_inception = False ))
testloader = torch.utils.data.DataLoader( testset, batch_size =128, shuffle = False,
num_workers = 8, pin_memory = True,  drop_last = False  )
# NOTE - if using batch size > 1 , model should be nn.module or similar capable of taking in a batched input

recall, nmi = get_recall_and_NMI(model, testloader )
"""
print("")

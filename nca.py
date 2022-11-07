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

datasetName = {
    "cub":CUBDataset,
    "sop":SOPDataset,
    "car":CARDataset
}["cub"]    # Change the "cub" on this line to select the correct dataset

latentDims = 20

nEpochs = 10
fileToSaveA = "Avalue.pickle"

dataset = datasetName()
params = {"batch_size":10, "num_workers":8, "shuffle":True}
loader = torch.utils.data.DataLoader(dataset, **params)

dev = "cuda" if torch.cuda.is_available() else "cpu"

h, w = dataset.getOutputShape()
A = nn.Parameter(torch.eye(latentDims, h*w*3, requires_grad=True))

optim = torch.optim.Adam([A])
for epoch in range(nEpochs):
    for images in tqdm.tqdm(loader):
        images = images.to(dev)
        optim.zero_grad()
        loss = computeLoss(A, images)
        loss.backward()
        optim.step()
    print(f"Done with {epoch+1}/{nEpochs} epochs")
torch.save({"A":A}, fileToSaveA)


testset= eval_dataset.load(name = ,  root = './data/', mode = 'eval',
transform = dataset.utils.make_transform( is_train = False, is_inception = False ))
testloader = torch.utils.data.DataLoader( testset, batch_size =128, shuffle = False,
num_workers = 8, pin_memory = True,  drop_last = False  )
# NOTE - if using batch size > 1 , model should be nn.module or similar capable of taking in a batched input

recall, nmi = get_recall_and_NMI(model, testloader )

import torch
import torchvision
import torch.nn as nn
import numpy as np
import tqdm
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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


data_path = './uspsdata/' # digit dataset from the paper
if not os.path.exists(data_path):
    os.makedirs(data_path)


batch_size = 64
latentDims = 32
nEpochs = 100

train_data = torchvision.datasets.USPS('./uspsdata/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

test_data = torchvision.datasets.USPS('./uspsdata/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"

h, w = next(iter(train_loader))[0].shape[-2:]

X_train = train_data.data.reshape(-1, h*w)
y_train = train_data.targets
X_test = test_data.data.reshape(-1, h*w)
y_test = test_data.targets

a = torch.randn(h*w, h*w) * 0.01  #full A
A = nn.Parameter(a)

optim = torch.optim.Adam([A], lr=1e-04)

for epoch in range(nEpochs):
    running_loss = 0

    for images, labels in train_loader:
        
        optim.zero_grad()
        
        X = images.reshape((-1, h*w))
        y_mask = labels[:, None] == labels[None, :] # pairwise boolean class matrix

        loss = computeLoss(A, X, y_mask)

        loss.backward()

        optim.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{nEpochs}: {running_loss * batch_size / len(train_data) :.4f}")

# Find the train and test accuracies
A = A.detach().cpu().numpy()

X_train_embed = X_train @ A.T
X_test_embed = X_test @ A.T
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train_embed, y_train)
train_pred = knn.predict(X_train_embed)
test_pred = knn.predict(X_test_embed)

train_acc = accuracy_score(train_pred, y_train)
test_acc = accuracy_score(test_pred, y_test)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")



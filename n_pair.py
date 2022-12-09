import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from deep_net.googlenet import *
from tqdm import *
from n_pair_loss import *
from pdb import set_trace as breakpoint
import eval_dataset
from evaluate import *

embed_size = 512
num_epochs = 60
lr = 1e-4
fc_lr = 5e-4
weight_decay = 1e-4
lr_decay_step = 10
lr_decay_gamma = 0.5
test_interval = 5
n_pair_l2_reg = 0.001


ALLOWED_MINING_OPS = ['npair']
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False


trainset = eval_dataset.load(name = "cub",  root = './data/CUB/', mode = 'train', transform = eval_dataset.utils.make_transform())
train_loader = torch.utils.data.DataLoader(trainset, batch_size = 100, shuffle = True, num_workers = 8, drop_last = True)

testset= eval_dataset.load(name = 'cub',  root = './data/CUB/', mode = 'eval', transform = eval_dataset.utils.make_transform( is_train = False))
test_loader = torch.utils.data.DataLoader( testset, batch_size =100, shuffle = False, num_workers = 8, pin_memory = True,  drop_last = False  )



dev = "cuda" if torch.cuda.is_available() else "cpu"

# CAN ADD MORE MODELS
model = googlenet_metric(embed_size=embed_size)
model.to(dev)

model_params = [
    {'params': list(set(model.parameters()).difference(set(model.model.embed_fc.parameters())))},
    {'params': model.model.embed_fc.parameters(), 'lr':float(fc_lr) }]

optim = torch.optim.Adam(model_params, lr=float(lr), weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size= lr_decay_step, gamma = lr_decay_gamma)
criterion_npair = n_pair_loss(n_pair_l2_reg)
criterion_npair.to(dev)

losses_list = []
train_recall_list=[]
val_recall_list = []
train_nmi_list=[]
val_nmi_list = []
best_recall = -1

for epoch in range(num_epochs):
    for batch_idx , (images, labels) in tqdm(enumerate(train_loader)):
        model.train()

        embed_image = model(images.to(dev))
        loss = criterion_npair(embed_image, labels.to(dev))
        optim.zero_grad()
        loss.backward()
        optim.step()

        if(batch_idx %20 ==0):
            losses_list.append(loss.item())
            print("Loss:", loss.item())

    scheduler.step()
    if(epoch % test_interval == 0):
        recall, nmi = get_recall_and_NMI(model, test_loader )
        val_recall_list.append(recall)
        val_nmi_list.append(nmi)

        train_recall, train_nmi = get_recall_and_NMI(model, train_loader )
        train_recall_list.append(train_recall)
        train_nmi_list.append(train_nmi)


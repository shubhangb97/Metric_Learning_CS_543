import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from deep_net.googlenet import *
from tqdm import *
from pnca_loss import *
from pdb import set_trace as breakpoint
import eval_dataset
from evaluate import *
import os

def save_dict(path, whichDataset, losses_list, train_recall_list, val_recall_list , train_nmi_list, val_nmi_list, best_recall ):
    info_dict= {}
    info_dict['losses'] = losses_list
    info_dict['train_recall'] = train_recall_list
    info_dict['val_recall'] = val_recall_list
    info_dict['train_nmi'] = train_nmi_list
    info_dict['val_nmi'] = val_nmi_list
    info_dict['best_recall'] = best_recall
    if(not(os.path.exists(path) ) ):
        os.mkdir(path)
    torch.save(info_dict, path+'/'+whichDataset+'_info_dict_n_pair.log')

embed_size = 512
num_epochs = 30
lr = 1e-4
fc_lr = 5e-4
weight_decay = 1e-4
lr_decay_step = 10
lr_decay_gamma = 0.5
test_interval = 10
n_pair_l2_reg = 0.001


ALLOWED_MINING_OPS = ['npair']
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False

whichDataset = 'cub'#'cub' # Choose from cub, cars, or SOP (works if you downloaded data using datasets.py)
if(whichDataset =='cub'):
    n_classes = 100
elif(whichDataset == 'cars'):
    n_classes = 98
elif(whichDataset == 'SOP'):
    n_classes = 11318
else:
    print('Specify correct dataset name')
    quit()
save_model_dict_path = f'./n_pair_model_dict_{whichDataset}.pt'
info_save_path = './results'

trainset = eval_dataset.load(name=whichDataset,
                            root='./data/'+whichDataset.upper()+'/',
                            mode='train',
                            transform = eval_dataset.utils.make_transform())
train_loader = torch.utils.data.DataLoader(trainset, batch_size = 100,
                        shuffle = True, num_workers = 8, drop_last = True)

testset= eval_dataset.load(name=whichDataset,
                            root='./data/'+whichDataset.upper()+'/',
                            mode='eval',
                            transform = eval_dataset.utils.make_transform(is_train=False))
test_loader = torch.utils.data.DataLoader(testset, batch_size =100,
                        shuffle=False, num_workers=8, pin_memory=True,
                        drop_last=False)

dev = "cuda" if torch.cuda.is_available() else "cpu"

# CAN ADD MORE MODELS
model = googlenet_metric(embed_size=embed_size)
model.to(dev)

model_params = [
    {'params': list(set(model.parameters()).difference(set(model.model.embed_fc.parameters())))},
    {'params': model.model.embed_fc.parameters(), 'lr':float(fc_lr) }]
criterion_pnca = pnca_loss(n_classes, embed_size)
criterion_pnca.to(dev)
model_params.append({'params': criterion_pnca.parameters(), 'lr':float(lr)})

optim = torch.optim.Adam(model_params, lr=float(lr), weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size= lr_decay_step, gamma = lr_decay_gamma)

losses_list = []
train_recall_list=[]
val_recall_list = []
train_nmi_list=[]
val_nmi_list = []
best_recall = -1

for epoch in range(num_epochs):
    for batch_idx , (images, labels) in enumerate(tqdm(train_loader)):
        model.train()

        embed_image = model(images.to(dev))
        loss = criterion_pnca(embed_image, labels.to(dev))
        #breakpoint()
        optim.zero_grad()
        loss.backward()
        optim.step()

        if(batch_idx %1000 ==0):
            losses_list.append(loss.item())
            print("Loss:", loss.item())

    scheduler.step()
    if(epoch % test_interval == 0):
        if(whichDataset == 'SOP'):
            recall, nmi = get_recall_SOP(model, test_loader )
            nmi = 0
        else:
            recall, nmi = get_recall_and_NMI(model, test_loader )
        val_recall_list.append(recall)
        val_nmi_list.append(nmi)

        if(whichDataset == 'SOP'):
            train_recall, train_nmi = get_recall_SOP(model, train_loader )
            train_nmi = 0
        else:
            train_recall, train_nmi = get_recall_and_NMI(model, train_loader )
        train_recall_list.append(train_recall)
        train_nmi_list.append(train_nmi)

        save_dict(info_save_path, whichDataset, losses_list, train_recall_list, val_recall_list , train_nmi_list, val_nmi_list, best_recall )
        torch.save(model.state_dict(), save_model_dict_path)

torch.save(model.state_dict(), save_model_dict_path)
print('\n\nFor the final model found:')
recall, nmi = get_recall_and_NMI(model, test_loader )
val_recall_list.append(recall)
val_nmi_list.append(nmi)

train_recall, train_nmi = get_recall_and_NMI(model, train_loader )
train_recall_list.append(train_recall)
train_nmi_list.append(train_nmi)

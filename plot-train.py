import numpy as np
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amsfonts}\usepackage{amssymb}')
import torch

whichDatasets = ['cub', 'cars']


for whichDataset in whichDatasets:
    infoFile = f'./results/{whichDataset}_info_dict_n_pair.log'
    data = torch.load(infoFile)
    #losses = data['losses']
    nmi = data['val_nmi']

    nEpochs = len(nmi)
    plt.plot(np.arange(0, nEpochs), nmi, label=f'{whichDataset.upper()} dataset')
plt.title(r'Accuracy as a function of epoch', fontsize='x-large')
plt.xlabel(r'Epochs trained', fontsize='x-large')
plt.ylabel(r'NMI', fontsize='x-large')
plt.legend(fontsize='x-large', framealpha=1.0)
plt.grid()
plt.savefig('over-time.pdf')


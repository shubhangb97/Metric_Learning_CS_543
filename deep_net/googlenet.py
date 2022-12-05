import torch, torch.nn as nn
import torchvision
class googlenet_metric(torch.nn.Module):
    def __init__(self, embed_size ):
        super(googlenet_metric, self).__init__()
        self.pars  = opt
        self.model = torchvision.models.googlenet(pretrained=True)
        self.model.embed_fc = torch.nn.Linear(self.model.fc.in_features, embed_size)
        self.model.fc = self.model.embed_fc

    def forward(self, x):
        x = self.model(x)
        return x

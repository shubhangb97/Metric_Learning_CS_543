
import torch, torch.nn as nn, torch.nn.functional as F
#import pretrainedmodels as ptm
import torchvision


"""============================================================="""
class bn_inception(torch.nn.Module):
    def __init__(self, embedding_size , return_embed_dict=False):
        super(bn_inception, self).__init__()

        #self.pars  = opt
        self.model = torchvision.models.inception_v3(pretrained= True)#ptm.__dict__['bninception'](num_classes=1000, pretrained='imagenet')
        self.model.fc = torch.nn.Linear(2048,embedding_size)
        # if '_he' in opt.arch:
        #     torch.nn.init.kaiming_normal_(self.model.last_linear.weight, mode='fan_out')
        #     torch.nn.init.constant_(self.model.last_linear.bias, 0)
        #
        # if 'frozen' in opt.arch:
        #     for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
        #         module.eval()
        #         module.train = lambda _: None

        # self.return_embed_dict = return_embed_dict
        #
        # self.pool_base = torch.nn.AdaptiveAvgPool2d(1)
        # self.pool_aux = torch.nn.AdaptiveMaxPool2d(1) #if 'double' in opt.arch else None
        #
        # self.name = 'bn_inception'#opt.arch
        #
        # self.out_adjust = None

    def forward(self, x, warmup=False, **kwargs):
        z = self.model(x)
        #y = self.pool_base(x)
        # if self.pool_aux is not None:
        #     y += self.pool_aux(x)
        # # if warmup:
        #     y,x = y.detach(), x.detach()
        # z = self.model.embed_fc(y.view(len(x),-1))
        # if 'normalize' in self.name:
        #     z = F.normalize(z, dim=-1)
        # if self.out_adjust and not self.training:
        #     z = self.out_adjust(z)
        return z

from .cars import *
from .cub import *
from .SOP import *
from .import utils
from .base import BaseDataset

# Based on code from https://github.com/tjddus9597/Proxy-Anchor-CVPR2020
# only for testing implementation of eval scripts
_type = {
    'cars': Cars,
    'cub': CUBirds,
    'SOP': SOP
}

def load(name, root, mode, transform = None):
    return _type[name](root = root, mode = mode, transform = transform)

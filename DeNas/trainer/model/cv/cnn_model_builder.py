import os
import sys
import ast
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from trainer.ModelBuilder import BaseModelBuilder
from cv.third_party.ZenNet import DeMainNet

class CNNModelBuilder(BaseModelBuilder):
    def __init__(self, args):
        
        self.args = args
    def init_model(self, args):
        return DeMainNet
    def create_model(self, arch, ext_dist):
        super_net = self.init_model(self.args)
        model = super_net(num_classes=self.args.num_classes, plainnet_struct=arch, no_create=False)
        if ext_dist.my_size > 1:
            model_dist = ext_dist.DDP(model, find_unused_parameters=True)
            return model_dist
        else:
            return model




            
            
            
        
            
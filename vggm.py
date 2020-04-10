#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:52:25 2020

@author: darp_lord
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from torchsummary import summary


class VGGM(nn.Module):
    
    def __init__(self, n_classes=1251):
        super(VGGM, self).__init__()
        self.n_classes=n_classes
        self.features=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(7,7), stride=(2,2), padding=1)),
            ('relu1', nn.ReLU()),
            ('mpool1', nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))),
            ('conv2', nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5), stride=(2,2), padding=1)),
            ('relu2', nn.ReLU()),
            ('mpool2', nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))),
            ('conv3', nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3), stride=(1,1), padding=1)),
            ('conv4', nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)),
            ('conv5', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)),
            ('mpool5', nn.MaxPool2d(kernel_size=(5,3), stride=(3,2))),
            ('fc6', nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=(9,1), stride=(1,1))),
            ('apool6', nn.AdaptiveAvgPool2d((1,1)))]))
            
        self.classifier=nn.Sequential(OrderedDict([
            ('fc7', nn.Linear(4096, 1024)),
            ('fc8', nn.Linear(1024, n_classes))]))
    
    def forward(self, inp):
        inp=self.features(inp)
        inp=inp.view(inp.size(0),-1)
        inp=self.classifier(inp)
        return inp

if __name__=="__main__":
    model=VGGM(1251)
    print(summary(model, (1,512,300)))
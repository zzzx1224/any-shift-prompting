import os
import sys
import time
import math
import random
import torch.nn as nn
import torch.nn.init as init
import torch
from torchvision import models
import numpy as np 
import pdb
import torch.nn.functional as f
# from main import args

resnet18 = models.resnet18(pretrained=True)
# resnet34 = models.resnet34(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
# resnet101 = models.resnet101(pretrained=True)
# alexnet = models.alexnet(pretrained=True)


class encoder(nn.Module):
    def __init__(self, backbone):
        super(encoder, self).__init__()

        self.backbone = backbone
        # backbone
        if backbone=='res18':
            resnet = resnet18
            self.feature_dim = 512
        elif backbone=='res50':
            resnet = resnet50
            self.feature_dim = 2048

        self.layer0 = nn.Sequential(
                        resnet.conv1,
                        resnet.bn1,
                        resnet.relu,
                        resnet.maxpool
                        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.pool = resnet.avgpool

        # common classifier
        if backbone=='res50':
            self.proj_layer = nn.Linear(self.feature_dim, 1024)

    def forward(self, x):
        # pdb.set_trace()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)

        if self.backbone == 'res50':
            x = self.proj_layer(x)

        return x
# Pytorch model of DeeplabV2resnet
# converted from caffe model to pth using https://github.com/kazuto1011/deeplab-pytorch#caffemodel
# adapted the model code from https://github.com/kazuto1011/deeplab-pytorch/blob/master/libs/models/deeplabv2.py

from collections import OrderedDict

import torch
import torch.nn as nn

from .resnet import _ConvBatchNormReLU, _ResBlock

import os

models_dir = os.path.abspath(os.path.dirname(__file__))


class DeepLabV2Stripped(nn.Sequential):
    """DeepLab v2 stripped of classification layer (Atrous Spatial Pyramid Pooling layer)"""

    def __init__(self, n_blocks):
        super().__init__()
        self.add_module(
            'fe_layer1',
            nn.Sequential(
                OrderedDict([
                    ('conv1', _ConvBatchNormReLU(3, 64, 7, 2, 3, 1)),
                    ('pool', nn.MaxPool2d(3, 2, 1, ceil_mode=True)),
                ])
            )
        )
        self.add_module('fe_layer2', _ResBlock(n_blocks[0], 64, 64, 256, 1, 1))
        self.add_module('fe_layer3', _ResBlock(n_blocks[1], 256, 128, 512, 2, 1))
        self.add_module('fe_layer4', _ResBlock(n_blocks[2], 512, 256, 1024, 1, 2))
        self.add_module('fe_layer5', _ResBlock(n_blocks[3], 1024, 512, 2048, 1, 4))

    def forward(self, x):
        return super().forward(x)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


if __name__ == '__main__':
    model = DeepLabV2Stripped(n_blocks=[3, 4, 23, 3])
    model.freeze_bn()
    model.eval()

    image = torch.autograd.Variable(torch.randn(1, 3, 513, 513))
    print(model(image)[0].size())

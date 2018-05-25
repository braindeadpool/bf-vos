# Network as proposed in https://arxiv.org/pdf/1804.03131.pdf
# Blazingly Fast Video Object Segmentation with Pixel-Wise Metric Learning, Chen et al

import torch.nn as nn
import torch
from collections import OrderedDict
from .deeplabv2resnet import DeepLabV2Stripped


class BFVOSNet(nn.Module):
    def __init__(self, embedding_vector_dims=128):
        """

        :param embedding_vector_dims: Embedding vector dimensions
        """
        super().__init__()
        # This will initialize the DeepLabv2Resnet as feature extractor
        self.network = DeepLabV2Stripped(n_blocks=[3, 4, 23, 3])
        # TODO: Concatenate array of pixel numbers and frame numbers (spatial and temporal information)
        # Add the embedding head
        self.network.add_module('eh_layer1',
                                nn.Sequential(
                                    OrderedDict([
                                        ('conv1', nn.Conv2d(2048, embedding_vector_dims, 1, 1)),
                                        ('relu1', nn.ReLU()),
                                    ])
                                ))
        self.network.add_module('eh_layer2', nn.Conv2d(embedding_vector_dims, embedding_vector_dims, 1, 1))

    def forward(self, x):
        return self.network(x)

    def freeze_feature_extraction(self):
        for name, module in self.named_children():
            if name.find('fe_') != -1:
                for param in module.parameters():
                    param.requires_grad = False


if __name__ == "__main__":
    model = BFVOSNet()
    model.freeze_bn()
    model.eval()

    image = torch.autograd.Variable(torch.randn(1, 3, 512, 512))
    print(model(image)[0].size())

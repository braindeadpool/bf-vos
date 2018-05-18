# Network as proposed in https://arxiv.org/pdf/1804.03131.pdf
# Blazingly Fast Video Object Segmentation with Pixel-Wise Metric Learning, Chen et al

import torch.nn as nn
import torch
import torch.nn.functional as F
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

    def forward(self, a, p, n):
        embedding_a = self.network(a)
        embedding_p = self.network(p)
        embedding_n = self.network(n)
        distance_p = F.pairwise_distance(embedding_p, embedding_a, 2)
        distance_n = F.pairwise_distance(embedding_n, embedding_a, 2)
        return distance_p, distance_n, embedding_a, embedding_p, embedding_n


if __name__ == "__main__":
    model = BFVOSNet()
    model.freeze_bn()
    model.eval()

    image = torch.autograd.Variable(torch.randn(1, 3, 512, 512))
    print(model(image)[0].size())

# Network as proposed in https://arxiv.org/pdf/1804.03131.pdf
# Blazingly Fast Video Object Segmentation with Pixel-Wise Metric Learning, Chen et al

import torch.nn as nn
import torch
from collections import OrderedDict
from .deeplabv2resnet import DeepLabV2Stripped
from .utils import init_weights


class BFVOSNet(DeepLabV2Stripped):
    def __init__(self, embedding_vector_dims=128):
        """

        :param embedding_vector_dims: Embedding vector dimensions
        """
        super().__init__(n_blocks=[3, 4, 23, 3])
        # TODO: Concatenate array of pixel numbers and frame numbers (spatial and temporal information)
        # Add the embedding head
        self.add_module('eh_layer1',
                        nn.Sequential(
                            OrderedDict([
                                ('conv1', nn.Conv2d(2048, embedding_vector_dims, 1, 1)),
                                ('relu1', nn.ReLU()),
                            ])
                        ))
        self.add_module('eh_layer2', nn.Conv2d(embedding_vector_dims, embedding_vector_dims, 1, 1))
        init_weights(self)

    def forward(self, x):
        embedding = super().forward(x)
        normalized_embedding = embedding / embedding.pow(2).sum(1, keepdim=True).sqrt()
        return normalized_embedding

    def freeze_feature_extraction(self):
        for name, m in self.named_children():
            if name.find('fe_') != -1:
                for param in m.parameters():
                    param.requires_grad = False


if __name__ == "__main__":
    model = BFVOSNet()
    model.freeze_bn()
    model.eval()

    image = torch.autograd.Variable(torch.randn(1, 3, 512, 512))
    print(model(image)[0].size())

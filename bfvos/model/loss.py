# Loss function

import torch
from . import config


def single_embedding_loss(anchor_point, pool):
    """
    Compute the loss from a pool of embedding points to a single anchor embedding point;
    as per paper, this computes the first term in the summation in Eq (1);
    for $x^a \in A$, this computes $min_{x^P \in P} ||f(x^a)-f(x^p)||^2_2$
    :param anchor_point:
    :param pool:
    :return:
    """
    return torch.min(torch.sum(torch.pow(torch.sub(pool, anchor_point), 2), dim=1))


def distance_matrix(x, y):
    """
    Computes distance matrix between two sets of embedding points
    shamelessly simplified version of https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
    :param x: N x d tensor
    :param y: M x d tensor (M need not be same as N)
    :return: N x M distance matrix
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, min=0.0)


class MinTripletLoss(torch.nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self._alpha = alpha

    def forward(self, anchor_points, positive_pool, negative_pool):
        positive_distances = distance_matrix(anchor_points, positive_pool)
        negative_distances = distance_matrix(anchor_points, negative_pool)
        slack = torch.autograd.Variable(
            torch.tensor(anchor_points.size(0) * self._alpha, dtype=config.DEFAULT_DTYPE),
            requires_grad=False)
        return torch.min(positive_distances, 1)[0].sum() - torch.min(negative_distances, 1)[
            0].sum() + slack

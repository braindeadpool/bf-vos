# Loss function

import torch
import torch.nn.functional as F


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
    # differences = x.unsqueeze(1) - y.unsqueeze(0)
    # distances = torch.sum(differences * differences, -1)
    # return distances


def validation_loss(anchor_points, positive_pool, negative_pool):
    """
    Computes validation loss as fraction of triplets where (negative sample, anchor point) is closer than (positive sample, anchor point)
    :param anchor_points: Nxd tensor representing N anchor points
    :param positive_pool: Mxd tensor representing M positive pool points
    :param negative_pool: Lxd tensor representing L negative pool points
    :return: float validation loss
    """
    positive_distances = distance_matrix(anchor_points, positive_pool)  # N x M
    negative_distances = distance_matrix(anchor_points, negative_pool)  # N x L
    N, M = positive_distances.size()
    N, L = negative_distances.size()
    p_ = positive_distances.repeat(1, L)  # N x (M*L)
    n_ = negative_distances.repeat(1, M)  # N x (M*L)
    # For each anchor point, for each combination of positive, negative pair, count how many pairs exist
    # where the positive point is farther than negative point
    return torch.sum(torch.gt(p_, n_)).float() / float(N * M * L)


class MinTripletLoss(torch.nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self._alpha = alpha

    def forward(self, anchor_points, positive_pool, negative_pool):
        positive_distances = distance_matrix(anchor_points, positive_pool)
        negative_distances = distance_matrix(anchor_points, negative_pool)
        losses = F.relu(torch.min(positive_distances, 1)[0].sum() - torch.min(negative_distances, 1)[
            0].sum() + self._alpha)
        return losses.mean()

# Loss function

import torch


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

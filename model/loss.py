# Loss function

import torch


def triplet_min_loss(anchors, positives, negatives, alpha=1):
    loss = torch.Tensor()
    for anchor in anchors:
        loss += torch.min(positives - anchor) - torch.min(negatives - anchor) + alpha
    return loss

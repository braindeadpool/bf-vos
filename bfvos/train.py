import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import davis
from .model import network, loss, config

root_dir = os.path.dirname(__file__)
torch.set_default_tensor_type(config.DEFAULT_TENSOR_TYPE)
torch.set_default_dtype(config.DEFAULT_DTYPE)

# Model configuration
image_width = 50
image_height = 50
embedding_vector_dims = 128

# Training parameters
# batch_size = 1
num_epochs = 1
learning_rate = 1e-3
momentum = 0.1
use_cuda = False
num_anchor_sample_points = 256  # according to paper
alpha = 1  # slack variable for loss


def main():
    data_source = davis.DavisDataset(base_dir=os.path.join(root_dir, 'dataset', 'DAVIS'),
                                     image_size=(image_width, image_height), year=2016, phase='train',
                                     transform=davis.ToTensor())
    triplet_sampler = davis.TripletSampler(dataset=data_source, randomize=True)
    data_loader = DataLoader(dataset=data_source, batch_sampler=triplet_sampler)

    model = network.BFVOSNet(embedding_vector_dims=embedding_vector_dims)
    loss_fn = loss.MinTripletLoss(alpha=alpha)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        train(data_loader, model, loss_fn, optimizer)


def train(data_loader, model, loss_fn, optimizer):
    for idx, sample in enumerate(data_loader):
        sample_frames = torch.autograd.Variable(sample['image'])
        embeddings = model(sample_frames)

        embedding_a = embeddings[0]
        embedding_f1 = embeddings[1]
        embedding_f2 = embeddings[2]
        embedding_pool_points = torch.cat([embedding_f1, embedding_f2], 2)  # horizontally stacked frame1 and frame 2
        # embedding_a/p/n is of shape (128, w/8, h/8)

        # TODO: Randomly sample anchor points from anchor frame
        # For now, use all anchor points in the image
        anchor_points = torch.ByteTensor(sample['annotation'][0])  # all anchor points
        fg_anchor_indices = torch.nonzero(anchor_points)
        bg_anchor_indices = torch.nonzero(anchor_points == 0)

        # all_pool_points is a binary tensor of shape (w/8, h/8).
        # For any index in all_pool_points, if it 1 => it is a foreground pixel
        all_pool_points = torch.cat([sample['annotation'][1], sample['annotation'][2]], 1)
        fg_pool_indices = torch.nonzero(all_pool_points)
        bg_pool_indices = torch.nonzero(all_pool_points == 0)

        fg_embedding_a = torch.cat([embedding_a[:, x, y].unsqueeze(0) for x, y in fg_anchor_indices])
        bg_embedding_a = torch.cat([embedding_a[:, x, y].unsqueeze(0) for x, y in bg_anchor_indices])

        # Compute loss for all foreground anchor points
        # For foreground anchor points,
        # positive pool: all foreground points in all_pool_points
        # negative pool: all background points in all_pool_points
        fg_positive_pool = torch.cat([embedding_pool_points[:, x, y].unsqueeze(0) for x, y in fg_pool_indices])
        bg_positive_pool = torch.cat([embedding_pool_points[:, x, y].unsqueeze(0) for x, y in bg_pool_indices])

        fg_negative_pool = bg_positive_pool
        bg_negative_pool = fg_positive_pool

        fg_loss = loss_fn(fg_embedding_a, fg_positive_pool, fg_negative_pool)
        bg_loss = loss_fn(bg_embedding_a, bg_positive_pool, bg_negative_pool)

        final_loss = fg_loss + bg_loss
        print("Loss = {}".format(final_loss))
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        if idx == 2:
            break


if __name__ == "__main__":
    main()

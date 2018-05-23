import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from bfvos.dataset import davis
from bfvos.model import network, loss, config
import logging
import time
import json

# Logging setup
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Set paths
root_dir = os.path.join('bfvos')
model_dir = os.path.join(root_dir, 'model')
pretrained_dir = os.path.join(model_dir, 'pretrained')
deeplab_resnet_pre_trained_path = os.path.join(pretrained_dir, 'deeplabv2_resnet_pretrained_compatible.pth')
checkpoint_dir = os.path.join(model_dir, 'checkpoints')
config_save_dir = os.path.join(model_dir, 'configs')

if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
if not os.path.exists(config_save_dir):
    os.mkdir(config_save_dir)

# Intervals
log_interval = 10
checkpoint_interval = 10

# Pytorch configs
torch.set_default_tensor_type(config.DEFAULT_TENSOR_TYPE)
torch.set_default_dtype(config.DEFAULT_DTYPE)

# select which GPU, -1 if CPU
has_cuda = False  # default
gpu_id = 0
if torch.cuda.is_available():
    has_cuda = True
    logger.info('Using GPU: {} '.format(gpu_id))
device = torch.device("cuda:{}".format(gpu_id) if has_cuda else "cpu")

seed = None
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)

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

training_config = {
    'image_width': image_width,
    'image_height': image_height,
    'embedding_vector_dims': embedding_vector_dims,
    'num_epochs': num_epochs,
    'learning_rate': learning_rate,
    'momentum': momentum,
    'alpha': alpha,
    'device': str(torch.device)
}


def main():
    data_source = davis.DavisDataset(base_dir=os.path.join(root_dir, 'dataset', 'DAVIS'),
                                     image_size=(image_width, image_height), year=2016, phase='train',
                                     transform=davis.ToTensor())
    triplet_sampler = davis.TripletSampler(dataset=data_source, randomize=True)
    data_loader = DataLoader(dataset=data_source, batch_sampler=triplet_sampler)

    model = network.BFVOSNet(embedding_vector_dims=embedding_vector_dims)
    loss_fn = loss.MinTripletLoss(alpha=alpha)
    if has_cuda:
        model = model.cuda()
        loss_fn = loss_fn.cuda()
        loss_fn.to(device)
        logger.debug("Model and loss function moved to CUDA")
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Load pre-trained model
    if has_cuda:
        model.load_state_dict(
            torch.load(deeplab_resnet_pre_trained_path, map_location=lambda storage, loc: storage.cuda(gpu_id)))
    else:
        model.load_state_dict(torch.load(deeplab_resnet_pre_trained_path))
    logger.info("Loaded DeepLab ResNet from {}".format(deeplab_resnet_pre_trained_path))
    # Load to appropriate device and set to training mode
    model.to(device).train()

    # Train
    for epoch in range(num_epochs):
        logger.info("Epoch {}/{}".format(epoch + 1, num_epochs))
        train(data_loader, model, loss_fn, optimizer, epoch)

    # Save final model
    model.eval().cpu()
    save_model_filename = "epoch_{}_{}.model".format(num_epochs, str(time.time()).replace(" ", "_").replace(".", "_"))
    save_model_path = os.path.join(model_dir, save_model_filename)
    torch.save(model.state_dict(), save_model_path)
    logger.info("Model saved to {}".format(save_model_filename))

    training_config_save_path = os.path.join(config_save_dir, save_model_filename.replace('.model', '.json'))
    with open(training_config_save_path, 'w') as f:
        json.dump(training_config, f)
        logger.info("Training config saved to {}".format(training_config_save_path))


def train(data_loader, model, loss_fn, optimizer, epoch):
    agg_fg_loss = 0.
    agg_bg_loss = 0.
    for idx, sample in enumerate(data_loader):
        if has_cuda:
            # move input tensors to gpu
            sample['image'] = sample['image'].to(device=device, dtype=config.DEFAULT_DTYPE)
            sample['annotation'] = sample['annotation'].to(device=device)
        sample_frames = sample['image']
        embeddings = model(sample_frames)

        embedding_a = embeddings[0]
        embedding_f1 = embeddings[1]
        embedding_f2 = embeddings[2]
        embedding_pool_points = torch.cat([embedding_f1, embedding_f2], 2)  # horizontally stacked frame1 and frame 2
        # embedding_a/p/n is of shape (128, w/8, h/8)

        # TODO: Randomly sample anchor points from anchor frame
        # For now, use all anchor points in the image
        if has_cuda:
            anchor_points = torch.cuda.ByteTensor(sample['annotation'][0])  # all anchor points
        else:
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

        # Backpropagation
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        # Logging

        agg_fg_loss += fg_loss.item()
        agg_bg_loss += bg_loss.item()
        if (idx + 1) % log_interval == 0:
            logger.info("Epoch: {}, Batch: {}".format(epoch + 1, idx + 1))
            logger.info("Avg FG Loss: {}, Avg BG Loss: {}, Avg Total Loss: {}".format(agg_fg_loss / (idx + 1),
                                                                                      agg_bg_loss / (idx + 1),
                                                                                      (agg_fg_loss + agg_bg_loss) / (
                                                                                          idx + 1)))

        if (idx + 1) % checkpoint_interval == 0:
            model.eval().cpu()
            ckpt_filename = "ckpt_epoch_{}_batch_id_{}.pth".format(epoch + 1, idx + 1)
            ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)
            torch.save(model.state_dict(), ckpt_path)
            logger.info("Checkpoint saved at {}".format(ckpt_filename))
            model.to(device).train()


if __name__ == "__main__":
    main()

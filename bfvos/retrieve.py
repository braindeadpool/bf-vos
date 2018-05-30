import argparse
import json
import logging
import numpy as np
import os
import torch
from PIL import Image
from .model import network, config

from sklearn.neighbors import NearestNeighbors

# Model configs
# Logging setup
logging.basicConfig()
logger = logging.getLogger(__name__)

# Set paths
root_dir = os.path.join('bfvos')

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


def load_image(input_frame_path, image_dims):
    input_frame = np.asarray(Image.open(input_frame_path).convert("RGB").resize(image_dims))
    # normalize the image
    input_frame = (input_frame - input_frame.mean()) / input_frame.std()
    return input_frame


def load_mask(input_mask_path, image_dims):
    input_mask = np.asarray(Image.open(input_mask_path).convert("L").resize(image_dims)).astype(np.uint8)
    input_mask[input_mask > 0] = 1
    return input_mask


def retrieve(args):
    # Default config
    image_dims = [256, 256]
    embedding_vector_dims = 128
    if args.model_config_file is not None:
        with open(args.model_config_file) as f:
            model_config = json.load(f)
            image_dims = list(map(int, model_config['image_dims']))
            embedding_vector_dims = int(model_config['embedding_vector_dims'])

    reduced_image_dims = list(np.array(image_dims) // 8 + 1)

    sequence_frames_batch = []
    output_file_paths = []
    for input_frame_filename in os.listdir(args.sequence_dir):
        input_frame_path = os.path.join(args.sequence_dir, input_frame_filename)
        filename, ext = os.path.splitext(input_frame_filename)
        if os.path.isfile(input_frame_path) and ext in ['.jpeg', '.jpg', '.png']:
            sequence_frames_batch.append(load_image(input_frame_path, image_dims))
            output_file_paths.append(os.path.join(args.output_dir, filename + '_output' + ext))

    input_image = load_image(args.input_image_path, image_dims)
    input_mask = load_mask(args.input_mask_path, reduced_image_dims)

    # Initialize and load model with config
    model = network.BFVOSNet(embedding_vector_dims=embedding_vector_dims)

    if has_cuda:
        model = model.cuda()
        logger.debug("Model initialized and moved to CUDA")
        model.load_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda(gpu_id)))
    else:
        model.load_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage))
    logger.info("Loaded weights from {}".format(args.model_path))
    model.to(device).eval()

    # Convert input to tensor of batch embeddings
    num_frames = len(sequence_frames_batch)
    idx = 0
    batch_size = args.batch_size if args.batch_size > 0 else num_frames
    all_embeddings = torch.Tensor()
    while idx < num_frames:
        current_batch_tensor = torch.from_numpy(
            np.array(sequence_frames_batch[idx:idx + batch_size]).transpose((0, 3, 1, 2)))
        if has_cuda:
            current_batch_tensor = current_batch_tensor.to(device=device, dtype=config.DEFAULT_DTYPE)
        batch_embeddings = model(current_batch_tensor)
        idx += batch_size
        all_embeddings = torch.cat((all_embeddings, batch_embeddings))
        logger.info("Computed embeddings for batch from frames {}-{}".format(idx, idx + batch_size))

    # Convert input image (reference image) to embedding
    input_image_tensor = torch.unsqueeze(torch.from_numpy(np.array(input_image).transpose(2, 0, 1)), 0)
    input_image_embedding = model(input_image_tensor).numpy().reshape((-1, embedding_vector_dims))
    input_mask_flattened = input_mask.flatten()

    logger.info("Computed embedding vectors for reference image")

    # Build nearest neighbor search tree and fit it to reference pixels
    nns = NearestNeighbors(n_neighbors=args.k, algorithm='ball_tree').fit(input_image_embedding)

    all_embeddings = all_embeddings.numpy().reshape((-1, embedding_vector_dims))
    output_mask = np.zeros(all_embeddings.shape[0])

    distances, indices = nns.kneighbors(all_embeddings)
    logger.info("Computed nearest neighbors for all test images")

    # For each pixel in input sequence,
    # find k nearest neighbor in reference image and do majority voting to assign label
    output_mask[np.sum(input_mask_flattened[indices], axis=1) > args.k / 2] = 255.0
    output_mask = output_mask.reshape([num_frames] + reduced_image_dims + [1])

    # Save each output mask (segmentation results)
    for idx, output_path in enumerate(output_file_paths):
        Image.fromarray(output_mask[idx].astype('uint8').squeeze()).resize(image_dims, resample=Image.BILINEAR).save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-dir', type=str, required=True,
                        help='Path to directory containing input image sequence')
    parser.add_argument('--input-image-path', type=str, required=True,
                        help='Path to input RGB image')
    parser.add_argument('--input-mask-path', type=str, required=True,
                        help='Path to image containing mask for image to segmented')
    parser.add_argument('--k', type=int, default=1, help='k for nearest neighbor search')
    parser.add_argument('--model-config-file', type=str, default=None,
                        help='Path to model configuration json file for inference')
    parser.add_argument('--model-path', type=str, required=True, help='Path to pre-trained model weight')
    parser.add_argument('--batch-size', type=int, default=-1, help='Batch size for inference')
    parser.add_argument('--output-dir', type=str, default='./', help='Output directory to save segmentation masks')
    parser.add_argument('--verbose', action='store_true', help='Print debug messages')
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    with torch.no_grad():
        retrieve(args)


if __name__ == "__main__":
    main()

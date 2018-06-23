from bfvos.retrieve import *
import matplotlib.path as mplp

# Configs
model_path = os.path.join(root_dir, 'ckpt_epoch_3_batch_1050.pth')
batch_size = 10
k = 5


def segment(input_sequence_paths, input_image_path, mask_coords, output_dir):
    sequence_frames_batch = []
    output_file_paths = []
    for input_frame_path in input_sequence_paths:
        filename, ext = os.path.splitext(input_frame_path)
        if os.path.isfile(input_frame_path) and ext in ['.jpeg', '.jpg', '.png']:
            sequence_frames_batch.append(load_image(input_frame_path, image_dims))
            output_file_paths.append(os.path.join(output_dir, filename + '_output' + ext))

    input_image = np.asarray(Image.open(input_image_path).convert("RGB"))
    original_dims = input_image.size
    input_image = input_image.resize(image_dims)
    # normalize the image
    input_image = (input_image - input_image.mean()) / input_image.std()

    # create pixel-wise boolean mask from polygon coordinates
    input_mask = np.zeros((original_dims[0], original_dims[1]), dtype=np.uint8)
    mask_polygon = mplp.Path(mask_coords)
    boolean_grid = mask_polygon.contains_points(np.vstack(np.nonzero(input_mask == 0)).T)
    input_mask[boolean_grid.reshape((original_dims[0], original_dims[1]))] = 1
    input_mask = input_mask.reshape(original_dims[:2] + [1])

    # Initialize and load model with config
    model = network.BFVOSNet(embedding_vector_dims=embedding_vector_dims)

    if has_cuda:
        model = model.cuda()
        logger.debug("Model initialized and moved to CUDA")
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage.cuda(gpu_id)))
    else:
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    logger.info("Loaded weights from {}".format(model_path))
    model.to(device).eval()

    return batch_segment(sequence_frames_batch, input_image, input_mask, output_file_paths, model, batch_size, k)

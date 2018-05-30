# bf-vos
## Pytorch implementation of Blazingly Fast Video Object Segmentation with Pixel-Wise Metric Learning (Chen et al)

## Major Requirements
Python3+, PyTorch, Pillow, NumPy, Scikit-learn

## Installation
`pip3 install -r requirements.txt`

## Usage
### Training
```bash
usage: python -m bfvos.train [-h] [-i width height] [-e EMBEDDING_VECTOR_DIMS]
                [-l LOG_INTERVAL] [-c CHECKPOINT_INTERVAL]
                [--val-interval VAL_INTERVAL] [-b BATCH_SIZE]
                [--val-batch-size VAL_BATCH_SIZE] [-n NUM_EPOCHS]
                [-r LEARNING_RATE] [-m MOMENTUM] [-a ALPHA]
                [--num-val-batches NUM_VAL_BATCHES] [-f LOG_FILE] [-v]
                [--checkpoint-path CHECKPOINT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -i width height, --image-dims width height
                        Input image dimensions as <width>, <height>
  -e EMBEDDING_VECTOR_DIMS, --embedding-vector-dims EMBEDDING_VECTOR_DIMS
                        Embedding vector dimensions
  -l LOG_INTERVAL, --log-interval LOG_INTERVAL
  -c CHECKPOINT_INTERVAL, --checkpoint-interval CHECKPOINT_INTERVAL
  --val-interval VAL_INTERVAL
                        Iterations after which to evaluate validation set
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Number of triplets in each batch
  --val-batch-size VAL_BATCH_SIZE
                        Numbed of triplets in each validation set batch
  -n NUM_EPOCHS, --num-epochs NUM_EPOCHS
  -r LEARNING_RATE, --learning-rate LEARNING_RATE
  -m MOMENTUM, --momentum MOMENTUM
  -a ALPHA, --alpha ALPHA
                        slack variable for loss
  --num-val-batches NUM_VAL_BATCHES
                        number of validation batches to evaluate, set to -1 to
                        evaluate whole set
  -f LOG_FILE, --log-file LOG_FILE
                        path to log file, setting this will log all messages
                        to this file
  -v, --verbose         Print debug messages
  --checkpoint-path CHECKPOINT_PATH
                        Path to checkpoint file to resume training, otherwise
                        train from scratch
```

### Online Retrieval
```bash
usage: python -m bfvos.retrieve [-h] --sequence-dir SEQUENCE_DIR --input-image-path
                   INPUT_IMAGE_PATH --input-mask-path INPUT_MASK_PATH [--k K]
                   [--model-config-file MODEL_CONFIG_FILE] --model-path
                   MODEL_PATH [--batch-size BATCH_SIZE]
                   [--output-dir OUTPUT_DIR] [--verbose]

optional arguments:
  -h, --help            show this help message and exit
  --sequence-dir SEQUENCE_DIR
                        Path to directory containing input image sequence
  --input-image-path INPUT_IMAGE_PATH
                        Path to input RGB image
  --input-mask-path INPUT_MASK_PATH
                        Path to image containing mask for image to segmented
  --k K                 k for nearest neighbor search
  --model-config-file MODEL_CONFIG_FILE
                        Path to model configuration json file for inference
  --model-path MODEL_PATH
                        Path to pre-trained model weight
  --batch-size BATCH_SIZE
                        Batch size for inference
  --output-dir OUTPUT_DIR
                        Output directory to save segmentation masks
  --verbose             Print debug messages

```

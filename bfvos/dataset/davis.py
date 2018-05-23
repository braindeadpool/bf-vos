import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, sampler
from natsort import natsorted


class DavisDataset(Dataset):
    """
    DAVIS dataset
    """

    def __init__(self, base_dir, image_size=None, year=None, phase='train', transform=None):
        """

        :param base_dir: path to DAVIS dataset directory
        :param image_size: (width, height) tuple to resize the image
        :param year: which train/val split of DAVIS to use
        :param phase: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        if phase in ['train', 'val', 'trainval']:
            self._base_dir = os.path.join(self._base_dir, 'trainval')
        elif phase in ['test', 'testdev']:
            self._base_dir = os.path.join(self._base_dir, 'testdev')
        elif phase == 'testchallenge':
            self._base_dir = os.path.join(self._base_dir, 'testchallenge')
        self._image_size = image_size
        self._images_dir = os.path.join(self._base_dir, 'JPEGImages', '480p')
        self._annotations_dir = os.path.join(self._base_dir, 'Annotations', '480p')
        self._transform = transform
        if year is not None:
            assert (year == 2016 or year == 2017)
            years = [year]
        else:
            years = [2016, 2017]
        self.sequences = []
        for year in years:
            with open(os.path.join(self._base_dir, 'ImageSets', '{}'.format(year), '{}.txt'.format(phase)), 'r') as f:
                self.sequences += [seq.strip() for seq in f.readlines()]
        self.sequences = sorted(list(set(self.sequences)))

        # Store all the image paths, annotation paths, frame numbers and sequence labels
        self._frame_data = []
        for seq in self.sequences:
            self._frame_data += list(map(lambda x: (
                os.path.join(self._images_dir, seq, x),
                os.path.join(self._annotations_dir, seq, x.replace('.jpg', '.png')),
                int(os.path.splitext(x)[0]), seq), natsorted(os.listdir(os.path.join(self._images_dir, seq)))))

        # Easy access to sequence specific framedata
        self.sequence_to_sample_idx = {seq: [] for seq in self.sequences}
        for idx, sample in enumerate(self._frame_data):
            self.sequence_to_sample_idx[sample[3]].append(idx)

    def __len__(self):
        return len(self._frame_data)

    def __getitem__(self, idx):
        # Load the <idx>th image and annotation and return a sample
        image_path, annotation_path, frame_no, label = self._frame_data[idx]
        image = Image.open(image_path).convert('RGB')
        annotation = Image.open(annotation_path).convert('L')
        if self._image_size is not None:
            image = image.resize(self._image_size)
            annotation = annotation.resize((self._image_size[0] // 8 + 1, self._image_size[1] // 8 + 1))

        # normalize the image
        image = np.asarray(image)
        image = (image - image.mean()) / image.std()

        # convert annotation to binary image
        annotation = np.asarray(annotation).copy()
        annotation[annotation > 0] = 1
        sample = {
            'image': image,
            'annotation': annotation,
            'frame_no': frame_no,
            'label': label
        }
        if self._transform:
            sample = self._transform(sample)
        return sample


class TripletSampler(sampler.Sampler):
    def __init__(self, dataset, sequence=None, randomize=True):
        super().__init__(data_source=dataset)
        self._dataset = dataset
        self._randomize = randomize
        if sequence is not None:
            self._num_samples = len(self._dataset.sequence_to_sample_idx[sequence])
            self._sequences = [sequence]
        else:
            self._num_samples = len(self._dataset)
            self._sequences = list(self._dataset.sequence_to_sample_idx.keys())

    def __iter__(self):
        for seq in self._sequences:
            # Now create set of random (a, p, n) samples from this sequence
            seq_sample_idx = self._dataset.sequence_to_sample_idx[seq]
            for i, a in enumerate(seq_sample_idx):
                # Randomly sample two non-anchor separate frames - one for positive pool and one for negative
                p, n = np.random.choice(seq_sample_idx[:i] + seq_sample_idx[i + 1:], size=2, replace=False)
                yield (a, p, n)

    def __len__(self):
        return self._num_samples


# Transforms
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return {'image': torch.from_numpy(sample['image'].transpose((2, 0, 1))),
                'annotation': torch.from_numpy(sample['annotation'].astype(np.uint8)),
                'frame_no': sample['frame_no'],
                'label': sample['label']
                }

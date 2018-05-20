import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from natsort import natsorted


class DavisDataset(Dataset):
    """
    DAVIS dataset
    """

    def __init__(self, base_dir, image_size=None, year=None, phase='train'):
        """

        :param base_dir: path to DAVIS dataset directory
        :param image_size: (width, height) tuple to resize the image
        :param year: which train/val split of DAVIS to use
        :param phase: train/val
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_size = image_size
        self._images_dir = os.path.join(self._base_dir, 'JPEGImages', '480p')
        self._annotations_dir = os.path.join(self._base_dir, 'Annotations', '480p')
        if year is not None:
            assert (year is 2016 or year is 2017)
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

    def __len__(self):
        return len(self._frame_data)

    def __getitem__(self, idx):
        # Load the <idx>th image and annotation and return a sample
        image_path, annotation_path, frame_no, label = self._frame_data[idx]
        image = Image.open(image_path).convert('RGB')
        annotation = Image.open(annotation_path).convert('L')
        if self._image_size is not None:
            image = image.resize(self._image_size)
            annotation = annotation.resize(self._image_size)

        # normalize the image
        image = np.asarray(image)
        image = (image - image.mean()) / image.std()

        # convert annotation to binary image
        annotation = np.asarray(annotation).copy()  # .copy() since original np.asarray(Image) object is read-only
        annotation[annotation > 0] = 1
        sample = {
            'image': image,
            'annotation': annotation,
            'frame_no': frame_no,
            'label': label
        }
        return sample

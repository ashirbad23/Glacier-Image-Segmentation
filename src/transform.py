import torch
import numpy as np
import random
from scipy.ndimage import rotate


class GlacierAugment:
    def __init__(self, flip_prob=0.5, rotate_prob=0.5, max_angle=45):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.max_angle = max_angle

    def __call__(self, bands, label):
        bands = bands.astype(np.float32)
        for i in range(bands.shape[0]):
            band = bands[i]
            bands[i] = (band - band.min()) / (band.max() - band.min() + 1e-6)
        label = label.astype(np.float32) / 255.0

        if random.random() < self.flip_prob:
            bands = np.flip(bands, axis=2)  #flip width
            label = np.flip(label, axis=1)

        if random.random() < self.flip_prob:
            bands = np.flip(bands, axis=1)  #flip height
            label = np.flip(label, axis=0)

        if random.random() < self.rotate_prob:
            angle = random.uniform(-self.max_angle, self.max_angle)
            bands_rot = np.zeros_like(bands)
            for i in range(bands.shape[0]):
                bands_rot[i] = rotate(bands[i], angle, reshape=False, order=1, mode='reflect')
            label = rotate(label, angle, reshape=False, order=0, mode='reflect')
            bands = bands_rot

        bands = bands.copy()
        label = label.copy()

        return torch.tensor(bands, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

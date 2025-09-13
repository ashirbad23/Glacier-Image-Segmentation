import torch
import numpy as np
import random


class GlacierAugment:
    def __init__(self, flip_prob=0.5, rotate_prob=0.5):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob

    def __call__(self, bands, label):
        bands = bands / 65535.0
        label = label.astype(np.float32) / 255.0

        if random.random() < self.flip_prob:
            bands = np.flip(bands, axis=2)  #flip width
            label = np.flip(label, axis=1)

        if random.random() < self.flip_prob:
            bands = np.flip(bands, axis=1)  #flip height
            label = np.flip(label, axis=0)

        if random.random() < self.rotate_prob:
            k = random.choice([1, 2, 3])
            bands = np.rot90(bands, k, axes=(1, 2))
            label = np.rot90(label, k)

        bands = bands.copy()
        label = label.copy()

        return torch.tensor(bands, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

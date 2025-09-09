import os
import matplotlib.pyplot as plt
from dataset import GlacierDataset
from transform import GlacierAugment  # your augmentation class
import numpy as np

base_path = "../data/Train"
band_folders = os.listdir(base_path)[:-1]

# Initialize augmentation
augment = GlacierAugment(flip_prob=0.5, rotate_prob=0.5)

# Dataset with augmentation
dataset = GlacierDataset(base_path=base_path, band_folders=band_folders, transform=augment)

# Get original (no transform) and augmented sample
dataset_raw = GlacierDataset(base_path=base_path, band_folders=band_folders, transform=None)

bands_raw, label_raw = dataset_raw[0]
bands_aug, label_aug = dataset[0]

# Plot comparison
plt.figure(figsize=(20, 6))

for i in range(5):
    # Original
    plt.subplot(2, 6, i+1)
    plt.imshow(bands_raw[i], cmap='gray')
    plt.title(f"Raw Band {i+1}")
    plt.axis('off')

    # Augmented
    plt.subplot(2, 6, i+7)
    plt.imshow(bands_aug[i].numpy(), cmap='gray')
    plt.title(f"Aug Band {i+1}")
    plt.axis('off')

# Plot labels
plt.subplot(2, 6, 6)
plt.imshow(label_raw, cmap='gray')
plt.title("Raw Label")
plt.axis('off')

plt.subplot(2, 6, 12)
plt.imshow(label_aug.numpy(), cmap='gray')
plt.title("Aug Label")
plt.axis('off')

plt.tight_layout()
plt.show()

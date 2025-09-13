import os
import matplotlib.pyplot as plt
from dataset import GlacierDataset
from transform import GlacierAugment
import numpy as np

base_path = "../data/Train"
band_folders = os.listdir(base_path)[:-1]

# Initialize augmentation
augment = GlacierAugment(flip_prob=0.5, rotate_prob=0.5)

# Datasets
dataset_raw = GlacierDataset(base_path=base_path, band_folders=band_folders, transform=None)
dataset_aug = GlacierDataset(base_path=base_path, band_folders=band_folders, transform=augment)

# Get one sample
bands_raw, label_raw = dataset_raw[0]
bands_aug, label_aug = dataset_aug[0]


# --- Helper to make RGB composite ---
def make_rgb(bands, indices=(2, 1, 0)):  # B4, B3, B2 indices
    rgb = np.stack([bands[i] for i in indices], axis=-1)  # Shape: H,W,3
    # Normalize for display
    rgb = rgb.astype(np.float32)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return rgb


# Prepare composites
rgb_raw = make_rgb(bands_raw)
rgb_aug = make_rgb(bands_aug.numpy())

# --- Plotting ---
plt.figure(figsize=(20, 10))

# 1. Raw bands
for i in range(5):
    plt.subplot(3, 6, i + 1)
    plt.imshow(bands_raw[i], cmap='gray')
    plt.title(f"Raw Band {i + 1}")
    plt.axis('off')

# 2. Aug bands
for i in range(5):
    plt.subplot(3, 6, i + 7)
    plt.imshow(bands_aug[i].numpy(), cmap='gray')
    plt.title(f"Aug Band {i + 1}")
    plt.axis('off')

# 3. Labels
plt.subplot(3, 6, 6)
plt.imshow(label_raw, cmap='gray')
plt.title("Raw Label")
plt.axis('off')

plt.subplot(3, 6, 12)
plt.imshow(label_aug.numpy(), cmap='gray')
plt.title("Aug Label")
plt.axis('off')

# 4. RGB Composites
plt.subplot(3, 6, 13)
plt.imshow(rgb_raw)
plt.title("Raw RGB (B4,B3,B2)")
plt.axis('off')

plt.subplot(3, 6, 14)
plt.imshow(rgb_aug)
plt.title("Aug RGB (B4,B3,B2)")
plt.axis('off')

plt.tight_layout()
plt.show()

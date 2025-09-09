import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class GlacierDataset(Dataset):
    def __init__(self, base_path: str, band_folders: list = None, transform=None, normalize=False):
        self.base_path = base_path
        self.band_folders = band_folders if band_folders else os.listdir(base_path)[:-1]
        self.labels_folder = os.listdir(base_path)[-1]
        self.transform = transform
        self.normalize = normalize  # <<=== New parameter da

        # Extract image IDs from the first band folder
        self.image_ids = [fname.split("_")[-2] + "_" + fname.split("_")[-1].split(".")[0]
                          for fname in os.listdir(os.path.join(self.base_path, self.band_folders[0]))]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        bands = []
        # Read all band images using OpenCV
        for folder in self.band_folders:
            band_dir = os.path.join(self.base_path, folder)
            band_file = [f for f in os.listdir(band_dir) if image_id in f][0]
            band_path = os.path.join(band_dir, band_file)
            img = cv2.imread(band_path, cv2.IMREAD_UNCHANGED)  # 16-bit unchanged
            bands.append(img)
        bands = np.stack(bands, axis=0).astype(np.float32)

        # Normalize if required
        if self.normalize:
            bands = bands / 65535.0

        # Read label using OpenCV
        label_dir = os.path.join(self.base_path, self.labels_folder)
        label_file = [f for f in os.listdir(label_dir) if image_id in f][0]
        label_path = os.path.join(label_dir, label_file)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        # Keep original labels for now (0–255)
        # You can binarize later if needed
        label = label.astype(np.float32)

        # Apply transform if given (e.g., augmentations)
        if self.transform:
            bands, label = self.transform(bands, label)

        return bands, label


if __name__ == "__main__":
    base_path = "../data/Train"
    band_folders = os.listdir(base_path)[:-1]

    # Toggle normalize on/off here
    dataset = GlacierDataset(base_path=base_path, band_folders=band_folders, normalize=False)

    # Test one sample
    bands, label = dataset[0]
    print("Bands shape:", bands.shape)  # [5, H, W]
    print("Label shape:", label.shape)  # [H, W]
    print("Bands max:", bands.max())
    print("Label unique:", np.unique(label))

    # Load dataset with and without normalization
    dataset_raw = GlacierDataset(base_path=base_path, band_folders=band_folders, normalize=False)
    dataset_norm = GlacierDataset(base_path=base_path, band_folders=band_folders, normalize=True)

    # Get same sample
    bands_raw, label_raw = dataset_raw[0]
    bands_norm, label_norm = dataset_norm[0]

    # Plot comparison for first 3 bands
    plt.figure(figsize=(15, 6))
    for i in range(5):
        # Raw band
        plt.subplot(2, 6, i + 1)
        plt.imshow(bands_raw[i], cmap='gray')
        plt.title(f"Raw Band {i + 1}\nMax: {bands_raw[i].max():.0f}")
        plt.axis('off')

        # Normalized band
        plt.subplot(2, 6, i + 7)
        plt.imshow(bands_norm[i], cmap='gray')
        plt.title(f"Norm Band {i + 1}\nMax: {bands_norm[i].max():.2f}")
        plt.axis('off')
    plt.subplot(2, 6, 6)
    plt.imshow(label_raw, cmap='grey')
    plt.title(f"Raw Label \n{label_raw.max()}")

    plt.subplot(2, 6, 12)
    plt.imshow(label_norm, cmap='grey')
    plt.title(f"Raw Label \n{label_norm.max()}")

    plt.suptitle("Raw vs Normalized Bands", fontsize=16)
    plt.tight_layout()
    plt.show()

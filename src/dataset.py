import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class GlacierDataset(Dataset):
    def __init__(self, base_path: str, band_folders: list = None, transform=None):
        self.base_path = base_path
        self.band_folders = band_folders if band_folders else os.listdir(base_path)[:-1]
        self.labels_folder = os.listdir(base_path)[-1]
        self.transform = transform

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

        # Read label using OpenCV
        label_dir = os.path.join(self.base_path, self.labels_folder)
        label_file = [f for f in os.listdir(label_dir) if image_id in f][0]
        label_path = os.path.join(label_dir, label_file)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        # Apply transform if given (e.g., augmentations)
        if self.transform:
            bands, label = self.transform(bands, label)

        return bands, label


if __name__ == "__main__":
    base_path = "../data/Train"
    band_folders = os.listdir(base_path)[:-1]

    # Toggle normalize on/off here
    dataset = GlacierDataset(base_path=base_path, band_folders=band_folders)

    # Test one sample
    bands, label = dataset[0]
    print("Bands shape:", bands.shape)  # [5, H, W]
    print("Label shape:", label.shape)  # [H, W]
    print("Bands max:", bands.max())
    print("Label unique:", np.unique(label))

    # Plot comparison for first 3 bands
    plt.figure(figsize=(15, 6))
    for i in range(5):
        # Raw band
        plt.subplot(1, 6, i + 1)
        plt.imshow(bands[i], cmap='gray')
        plt.title(f"Raw Band {i + 1}\nMax: {bands[i].max():.0f}")
        plt.axis('off')

    plt.subplot(1, 6, 6)
    plt.imshow(label, cmap='grey')
    plt.title(f"Raw Label \n{label.max()}")

    plt.tight_layout()
    plt.show()

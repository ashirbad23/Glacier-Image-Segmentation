import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from transform import GlacierAugment


class GlacierDataset(Dataset):
    def __init__(self, base_path: str, patch_size=256, for_ann=False):
        self.base_path = base_path
        self.band_folders = os.listdir(base_path)[:-1]
        self.labels_folder = os.listdir(base_path)[-1]
        self.patch_size = patch_size
        self.for_ann = for_ann
        self.transform = GlacierAugment()

        # Extract image IDs from the first band folder
        self.image_ids = [fname.split("_")[-2] + "_" + fname.split("_")[-1].split(".")[0]
                          for fname in os.listdir(os.path.join(self.base_path, self.band_folders[0]))]

        self.samples = self._generate_samples()

    def __len__(self):
        return len(self.samples)

    def _generate_samples(self):
        samples = []
        for img_id in self.image_ids:
            band_dir = os.path.join(self.base_path, self.band_folders[0])
            band_file = [f for f in os.listdir(band_dir) if img_id in f][0]
            band_path = os.path.join(band_dir, band_file)
            band = cv2.imread(band_path, cv2.IMREAD_UNCHANGED)  # 16-bit unchanged
            H, W = band.shape

            for y in range(0, H, self.patch_size):
                for x in range(0, W, self.patch_size):
                    for t_id in self.transform.transform_list:
                        samples.append((img_id, x, y, self.patch_size, self.patch_size, t_id))

        return samples

    def __getitem__(self, idx):
        img_id, x, y, h, w, t_id = self.samples[idx]

        bands = []
        # Read all band images using OpenCV
        for folder in self.band_folders:
            band_dir = os.path.join(self.base_path, folder)
            band_file = [f for f in os.listdir(band_dir) if img_id in f][0]
            band_path = os.path.join(band_dir, band_file)
            img = cv2.imread(band_path, cv2.IMREAD_UNCHANGED)  # 16-bit unchanged
            bands.append(img[y:y + h, x:x + w])
        bands = np.stack(bands, axis=0).astype(np.float32)

        # Read label using OpenCV
        label_dir = os.path.join(self.base_path, self.labels_folder)
        label_file = [f for f in os.listdir(label_dir) if img_id in f][0]
        label_path = os.path.join(label_dir, label_file)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        label = label[y:y + h, x:x + w]

        # Apply transform if given (e.g., augmentations)
        bands, label = self.transform.apply(bands, label, t_id)

        for i in range(bands.shape[0]):
            bmin, bmax = bands[i].min(), bands[i].max()
            bands[i] = (bands[i] - bmin) / (bmax - bmin + 1e-6)

        label = label / 255

        return torch.tensor(bands, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


class GlacierPixelDataset(Dataset):
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.band_folders = os.listdir(base_path)[:-1]
        self.labels_folder = os.listdir(base_path)[-1]

        self.X_pixels = []
        self.y_pixels = []

        # Preprocess all images
        for image_file in os.listdir(os.path.join(base_path, self.band_folders[0])):
            img_id = "_".join(image_file.split("_")[-2:]).split(".")[0]
            bands_list = []

            # Load all bands
            for folder in self.band_folders:
                band_dir = os.path.join(base_path, folder)
                band_file = [f for f in os.listdir(band_dir) if img_id in f][0]
                band_path = os.path.join(band_dir, band_file)
                img = cv2.imread(band_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                if img.ndim == 3:
                    img = img[..., 0]
                mean, std = np.mean(img), np.std(img)
                if std > 0:
                    img = (img - mean) / std
                bands_list.append(img.flatten())

            # Stack bands to shape (H*W, 5)
            X = np.stack(bands_list, axis=1)

            # Load label
            label_dir = os.path.join(base_path, self.labels_folder)
            label_file = [f for f in os.listdir(label_dir) if img_id in f][0]
            label_path = os.path.join(label_dir, label_file)
            y = cv2.imread(label_path, cv2.IMREAD_UNCHANGED).flatten().astype(np.float32) / 255

            # Remove cloud pixels
            mask = X.sum(axis=1) != 0
            self.X_pixels.append(X[mask])
            self.y_pixels.append(y[mask])

        # Concatenate all images
        self.X_pixels = np.vstack(self.X_pixels)
        self.y_pixels = np.hstack(self.y_pixels)

    def __len__(self):
        return self.X_pixels.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.X_pixels[idx], dtype=torch.float32), torch.tensor(self.y_pixels[idx],
                                                                                   dtype=torch.float32)


if __name__ == "__main__":
    base_path = "../data/Train"
    band_folders = os.listdir(base_path)[:-1]

    # Toggle normalize on/off here
    dataset = GlacierDataset(base_path=base_path)

    # Test one sample
    bands, label = dataset[880]
    bands, label = bands.numpy(), label.numpy()
    print("Bands shape:", bands.shape)  # [5, H, W]
    print("Label shape:", label.shape)  # [H, W]
    print("Bands max:", bands.max())
    print("Label unique:", np.unique(label))
    print(len(dataset))

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

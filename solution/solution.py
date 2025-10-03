# pip install torch torchvision numpy opencv-python matplotlib scikit-learn pillow

import os
import re
import cv2
import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """(Conv => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class UNetPP(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, deep_supervision=False, base_filters=32):
        super(UNetPP, self).__init__()
        self.deep_supervision = deep_supervision

        nb_filter = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8, base_filters * 16]

        # Encoder
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])

        self.pool = nn.MaxPool2d(2)

        # Decoder (nested)
        self.up1_0 = Up(nb_filter[1], nb_filter[0])
        self.up2_0 = Up(nb_filter[2], nb_filter[1])
        self.up3_0 = Up(nb_filter[3], nb_filter[2])
        self.up4_0 = Up(nb_filter[4], nb_filter[3])

        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[0], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[1], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[2], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[3], nb_filter[3])

        self.conv0_2 = ConvBlock(nb_filter[0] * 2 + nb_filter[0], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1] * 2 + nb_filter[1], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2] * 2 + nb_filter[2], nb_filter[2])

        self.conv0_3 = ConvBlock(nb_filter[0] * 3 + nb_filter[0], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1] * 3 + nb_filter[1], nb_filter[1])

        self.conv0_4 = ConvBlock(nb_filter[0] * 4 + nb_filter[0], nb_filter[0])

        # Deep supervision heads
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Decoder path (nested connections)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_0(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_0(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_0(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_0(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_0(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_0(x1_3)], 1))

        # Output
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output


def get_tile_id(filename):
    match = re.search(r"img(\d+)\.tif", filename)
    if match:
        return match.group(1)
    match = re.search(r"(\d{2}_\d{2})", filename)
    return match.group(1) if match else None


def maskgeration(image_path, model_path, patch_size=128, threshold=0.5, device='cpu'):
    model = UNetPP(in_channels=5, out_channels=1)
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)
    model.to(device).eval()

    # map band->tile->filename
    band_tile_map = {b: {} for b in image_path}
    for band, folder in image_path.items():
        for fname in os.listdir(folder):
            if fname.endswith(".tif"):
                tid = get_tile_id(fname)
                if tid:
                    band_tile_map[band][tid] = os.path.join(folder, fname)

    ref_band = sorted(image_path.keys())[0]
    tile_ids = sorted(band_tile_map[ref_band].keys())
    results = {}

    for tid in tile_ids:
        # load all bands for this tile
        band_arrays = []
        for band in sorted(image_path.keys()):
            arr = cv2.imread(band_tile_map[band][tid], cv2.IMREAD_UNCHANGED)
            band_arrays.append(arr.astype(np.float32))

        bands = np.stack(band_arrays, 0)  # shape [B,H,W]
        B, H, W = bands.shape

        # per-band min-max norm
        for b in range(B):
            mn, mx = bands[b].min(), bands[b].max()
            bands[b] = (bands[b] - mn) / (mx - mn + 1e-6)

        # prepare output mask
        mask = np.zeros((H, W), np.uint8)

        # iterate non-overlapping patches
        for y in range(0, H, patch_size):
            for x in range(0, W, patch_size):
                patch = bands[:, y:y + patch_size, x:x + patch_size]
                if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                    continue  # skip partial edges

                if patch.sum() == 0:
                    continue  # skip empty patches

                inp = torch.from_numpy(patch).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(inp)
                    if isinstance(out, (list, tuple)):
                        out = out[-1]
                    probs = torch.sigmoid(out).cpu().numpy()[0, 0]

                mask[y:y + patch_size, x:x + patch_size] = (probs >= threshold).astype(np.uint8) * 255

        results[tid] = mask

    return results


import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef


def mcc_score(y_true, y_pred, threshold=0.5):
    """Compute MCC after applying sigmoid + threshold."""
    y_true = y_true / 255
    y_pred = y_pred / 255
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    return matthews_corrcoef(y_true, y_pred)


def main():
    # Paths
    image_path = {
        "B1": "../data/Train/Band1",
        "B2": "../data/Train/Band2",
        "B3": "../data/Train/Band3",
        "B4": "../data/Train/Band4",
        "B5": "../data/Train/Band5",
    }
    model_path = "../weights/model_finetuned.pth"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch_size = 128
    threshold = 0.5

    mcc_total = 0

    # Run inference
    print("Running inference on test images...")
    preds = maskgeration(image_path, model_path, patch_size, threshold, device)

    # Save + visualize results
    for tid, mask in preds.items():
        out_path = os.path.join(output_dir, f"pred_{tid}.png")
        cv2.imwrite(out_path, mask)

        # Show side-by-side visualization
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        # Just show band B1 as reference
        labels = [img for img in os.listdir("../data/Train/label") if tid in img]
        ref_img_path = os.path.join("../data/Train/label", labels[0])
        ref_img = cv2.imread(ref_img_path, cv2.IMREAD_UNCHANGED)

        mcc = mcc_score(ref_img, mask)
        mcc_total += mcc
        ax[0].imshow(ref_img, cmap='gray')
        ax[0].set_title(f"Original {tid}")
        ax[0].axis("off")

        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title(f"Predicted Mask {tid}")
        ax[1].axis("off")

        ax[1].text(
            5, 15,  # x, y coordinates inside the axis
            f"MCC: {mcc:.2f}",  # Text
            color="red",
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.5)
        )

        plt.tight_layout()
        plt.savefig(f"Evaluation/{tid}.png")
        plt.show()

    print(f"Results saved to: {output_dir}")
    print(f"MCC: {mcc_total / 25}")


if __name__ == "__main__":
    main()

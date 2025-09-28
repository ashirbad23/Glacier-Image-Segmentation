import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef
import numpy as np
from loss import BCEDiceLoss
from model import UNetPP
from dataset import GlacierDataset
from tqdm import tqdm
import os

os.makedirs("../weights", exist_ok=True)
os.makedirs("../outputs", exist_ok=True)

EPOCHS = 100
BATCH_SIZE = 2
LR = 5e-5
PATIENCE = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PATH = "../data/Train"


def mcc_score(y_true, y_pred, threshold=0.5):
    """Compute MCC after applying sigmoid + threshold."""
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred.detach().cpu().numpy().ravel() > threshold).astype(np.int32)
    y_true = y_true.detach().cpu().numpy().ravel().astype(np.int32)
    return matthews_corrcoef(y_true, y_pred)


def train_one_loop(epoch, model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_mcc = 0
    pbar = tqdm(loader, desc=f'Epoch: {epoch}', leave=False)

    for bands, labels in pbar:
        bands, labels = bands.to(device), labels.to(device)
        labels = labels.unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(bands)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mcc += mcc_score(labels, outputs)

        pbar.set_postfix({
            "Batch loss": loss.item(),
            "Batch MCC": [round(mcc_score(labels, outputs), 4)]
        })

    return total_loss / len(loader), total_mcc / len(loader)


def main():
    dataset = GlacierDataset(base_path=BASE_PATH, patch_size=128)
    model = UNetPP(in_channels=5, out_channels=1)
    model.load_state_dict(torch.load('../weights/modelUNPP.pth', map_location=DEVICE))
    model = model.to(DEVICE)

    X = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )

    best_mcc = -1
    no_improve = 0

    for epoch in range(EPOCHS):
        train_loss, train_mcc = train_one_loop(epoch, model, X, optimizer, criterion, DEVICE)
        scheduler.step(train_loss)

        print(f"Epoch {epoch + 1} | Loss: {train_loss:.4f} | MCC: {train_mcc:.4f}")

        if train_mcc > best_mcc:
            best_mcc = train_mcc
            torch.save(model.state_dict(), '../weights/model_finetuned.pth')
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print("Early stopping triggered!")
            break


if __name__ == "__main__":
    main()

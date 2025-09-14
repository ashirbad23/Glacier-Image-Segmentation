import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef
import numpy as np
from dataset import GlacierDataset
from transform import GlacierAugment
from model import U_Net
from tqdm import tqdm
import pickle

# Hyperparameters
EPOCHS = 60
BATCH_SIZE = 2
LR = 1e-3
KFOLDS = 5
PATIENCE = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PATH = "../data/Train"


# Metrics MCC
def mcc_score(y_ture, y_pred):
    y_true = y_ture.cpu().numpy().flatten()
    y_pred = (y_pred.cpu().numpy().flatten() > 0.5).astype(np.int32)
    return matthews_corrcoef(y_true, y_pred)


# Training Loop
def train_one_loop(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc='Training', leave=False)
    for bands, labels in pbar:
        bands, labels = bands.to(device), labels.to(device)
        labels = labels.unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(bands)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"Batch loss": loss.item()})
    return total_loss / len(loader)


# Validation Loop
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_mcc = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for bands, labels in pbar:
            bands, labels = bands.to(device), labels.to(device)
            labels = labels.unsqueeze(1)
            outputs = model(bands)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_mcc += mcc_score(labels, outputs)
    return total_loss / len(loader), total_mcc / len(loader)


def main():
    dataset = GlacierDataset(base_path=BASE_PATH, transform=GlacierAugment())
    kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n===== FOLD {fold + 1} =====")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

        model = U_Net(in_channels=5, out_channels=1).to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        best_mcc = -1
        patience_counter = 0

        train_losses, val_losses = [], []
        train_mccs, val_mccs = [], []

        for epoch in range(EPOCHS):
            train_loss = train_one_loop(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_mcc = validate(model, val_loader, criterion, DEVICE)

            scheduler.step(val_loss)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | MCC: {val_mcc:.4f}")

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_mccs.append(val_mcc)

            if val_mcc > best_mcc:
                best_mcc = val_mcc
                torch.save(model.state_dict(), f"../weights/best_model_fold{fold}_{val_mcc}.pth")
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"===== Early stopping at epoch: {epoch} =====")

        with open(f"../outputs/metrics_fold{fold}.pkl", "wb") as f:
            pickle.dump({
                "train_loss": train_losses,
                "val_loss": val_losses,
                "val_mcc": val_mccs
            }, f)


if __name__ == "__main__":
    main()

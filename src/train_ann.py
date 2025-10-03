import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef
import numpy as np
import os
import pickle
from dataset import GlacierPixelDataset  # your dataset file
from loss import BCEDiceLossANN        # your combined BCE+Dice loss
from model import SpectralPixelNet
from tqdm import tqdm

# Folders for outputs
os.makedirs("../weights_ann", exist_ok=True)
os.makedirs("../outputs_ann", exist_ok=True)

# Seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# Hyperparameters
EPOCHS = 1000
BATCH_SIZE = 1024
LR = 1e-3
KFOLDS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PATH = "../data/Train"


# ----- Metrics -----
def mcc_score(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred.detach().cpu().numpy().ravel() > threshold).astype(np.int32)
    y_true = y_true.detach().cpu().numpy().ravel().astype(np.int32)
    return matthews_corrcoef(y_true, y_pred)


# ----- Training Loop -----
def train_one_loop(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_mcc = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for features, labels in pbar:
        features, labels = features.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mcc += mcc_score(labels, outputs)
        pbar.set_postfix({"Batch loss": loss.item(), "Batch MCC": mcc_score(labels, outputs)})
    return total_loss / len(loader), total_mcc / len(loader)


# ----- Validation Loop -----
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_mcc = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device).unsqueeze(1)
            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_mcc += mcc_score(labels, outputs)
    return total_loss / len(loader), total_mcc / len(loader)


# ----- Main -----
def main():
    dataset = GlacierPixelDataset(base_path=BASE_PATH)
    kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n===== FOLD {fold + 1} =====")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

        model = SpectralPixelNet(in_channels=5, out_channels=1).to(DEVICE)
        criterion = BCEDiceLossANN()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        best_mcc = -1
        best_model_path = None
        train_losses, val_losses, val_mccs = [], [], []

        for epoch in range(EPOCHS):
            train_loss, train_mcc = train_one_loop(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_mcc = validate(model, val_loader, criterion, DEVICE)

            scheduler.step(val_loss)

            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Train MCC: {train_mcc:.4f} "
                  f"| Val Loss: {val_loss:.4f} | Val MCC: {val_mcc:.4f}")

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_mccs.append(val_mcc)

            if val_mcc > best_mcc:
                best_mcc = val_mcc
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_model_path = f"../weights_ann/best_model_fold{fold}_{val_mcc:.4f}.pth"
                torch.save(model.state_dict(), best_model_path)

        # Save metrics per fold
        with open(f"../outputs_ann/metrics_fold{fold}.pkl", "wb") as f:
            pickle.dump({
                "train_loss": train_losses,
                "val_loss": val_losses,
                "val_mcc": val_mccs
            }, f)

        torch.cuda.empty_cache()
        del model, optimizer, criterion, train_loader, val_loader


if __name__ == "__main__":
    main()

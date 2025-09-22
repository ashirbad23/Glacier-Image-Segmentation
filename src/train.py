import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef
import numpy as np
from dataset import GlacierDataset
from loss import BCEDiceLoss
from model import UNetPP  # You can switch to U_Net if needed
from tqdm import tqdm
import pickle
import os

# =====================
# Setup
# =====================
os.makedirs("../weights", exist_ok=True)
os.makedirs("../outputs", exist_ok=True)

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 2
LR = 1e-3
KFOLDS = 5
PATIENCE = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_PATH = "../data/Train"


# =====================
# Metrics MCC
# =====================
def mcc_score(y_true, y_pred, threshold=0.5):
    """Compute MCC after applying sigmoid + threshold."""
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred.detach().cpu().numpy().ravel() > threshold).astype(np.int32)
    y_true = y_true.detach().cpu().numpy().ravel().astype(np.int32)
    return matthews_corrcoef(y_true, y_pred)


# =====================
# Training Loop
# =====================
def train_one_loop(model, loader, optimizer, criterion, device, thresholds):
    model.train()
    total_loss = 0
    total_mcc = np.zeros(len(thresholds))
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
        for i, thresh in enumerate(thresholds):
            total_mcc[i] += mcc_score(labels, outputs, threshold=thresh)

        pbar.set_postfix({
            "Batch loss": loss.item(),
            "Batch MCC": [round(mcc_score(labels, outputs, t), 4) for t in thresholds]
        })

    return total_loss / len(loader), total_mcc / len(loader)


# =====================
# Validation Loop
# =====================
def validate(model, loader, criterion, device, thresholds):
    model.eval()
    total_loss = 0
    total_mcc = np.zeros(len(thresholds))
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for bands, labels in pbar:
            bands, labels = bands.to(device), labels.to(device)
            labels = labels.unsqueeze(1)
            outputs = model(bands)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            for i, thresh in enumerate(thresholds):
                total_mcc[i] += mcc_score(labels, outputs, threshold=thresh)

    return total_loss / len(loader), total_mcc / len(loader)


# =====================
# Main Training
# =====================
def main():
    dataset = GlacierDataset(base_path=BASE_PATH)
    kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)

    thresholds = [0.3, 0.4, 0.5, 0.6]

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n===== FOLD {fold + 1} =====")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

        model = UNetPP(in_channels=5, out_channels=1).to(DEVICE)
        criterion = BCEDiceLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Track best results separately for each threshold
        best_mcc_per_thresh = {t: -1 for t in thresholds}
        best_model_path_per_thresh = {t: None for t in thresholds}

        patience_counter = 0

        train_losses, val_losses = [], []
        train_mccs, val_mccs = [], []

        for epoch in range(EPOCHS):
            train_loss, train_mcc = train_one_loop(model, train_loader, optimizer, criterion, DEVICE, thresholds)
            val_loss, val_mcc = validate(model, val_loader, criterion, DEVICE, thresholds)

            scheduler.step(val_loss)

            print(
                f"Epoch {epoch + 1} | "
                f"Train Loss: {train_loss:.4f} | Train MCC: {np.round(train_mcc, 4)} | "
                f"Val Loss: {val_loss:.4f} | Val MCC: {np.round(val_mcc, 4)}"
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_mccs.append(val_mcc)

            improved = False
            for i, t in enumerate(thresholds):
                if val_mcc[i] > best_mcc_per_thresh[t]:
                    best_mcc_per_thresh[t] = val_mcc[i]

                    # Remove old model for this threshold
                    if best_model_path_per_thresh[t] is not None and os.path.exists(best_model_path_per_thresh[t]):
                        os.remove(best_model_path_per_thresh[t])

                    # Save model for this threshold
                    best_model_path_per_thresh[t] = f"../weights/fold{fold + 1}_th{t:.2f}_mcc{np.round(val_mcc, 4)}.pth"
                    torch.save(model.state_dict(), best_model_path_per_thresh[t])
                    print(f"✅ New best model saved for fold {fold + 1}, threshold {t:.2f}, MCC {np.round(val_mcc, 4)}")
                    improved = True

            if improved:
                patience_counter = 0
            else:
                patience_counter += 1

            # if patience_counter >= PATIENCE:
            #     print(f"===== Early stopping at epoch {epoch+1} for fold {fold+1} =====")
            #     break

        # Save metrics for this fold
        with open(f"../outputs/metrics_fold{fold + 1}.pkl", "wb") as f:
            pickle.dump({
                "train_loss": train_losses,
                "val_loss": val_losses,
                "val_mcc": val_mccs
            }, f)

        torch.cuda.empty_cache()
        del model, optimizer, criterion, train_loader, val_loader


if __name__ == "__main__":
    main()

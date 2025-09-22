import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth
        self.sigmoid = nn.Sigmoid()

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)

        preds = self.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)

        return bce_loss + dice_loss


class BCEDiceLossANN(nn.Module):
    def __init__(self, smooth=1e-5):
        super(BCEDiceLossANN, self).__init__()
        self.bce = nn.BCELoss()
        self.smooth = smooth

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)

        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)

        return bce_loss + dice_loss

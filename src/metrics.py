import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def dice_score(pred, target, n_classes=3):
    preds = pred.argmax(dim=1)
    dice_sum = 0.0

    for cls in range(1, n_classes):
        pred_c = (preds == cls).float()
        target_c = (target == cls).float()

        inter = (pred_c * target_c).sum()
        dice = (2 * inter) / (pred_c.sum() + target_c.sum() + 1e-8)
        dice_sum += dice.item()

    return dice_sum / (n_classes - 1)


def classification_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

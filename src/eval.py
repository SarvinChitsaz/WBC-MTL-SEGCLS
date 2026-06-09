import torch
from sklearn.metrics import classification_report, confusion_matrix
from .metrics import dice_score, classification_metrics


def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []
    dice_total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)

            seg_out, cls_out = model(images)

            preds = cls_out.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            dice_total += dice_score(seg_out, masks)

    metrics = classification_metrics(all_labels, all_preds)
    metrics["dice"] = dice_total / len(loader)

    return metrics, all_labels, all_preds


def print_report(y_true, y_pred, target_names):
    print(classification_report(y_true, y_pred, target_names=target_names))
    print(confusion_matrix(y_true, y_pred))

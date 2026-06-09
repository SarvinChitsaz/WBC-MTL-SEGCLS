import torch
import torch.nn as nn
from configs.config import *
from data.dataset import get_dataloaders
from data.transforms import train_transform, test_transform
from models.model import UNetResNet34MultiTask


def main():

    base_path = DATA_DIR

    train_loader, test_loader, n_classes_cls, le = get_dataloaders(
        base_path=base_path,
        batch_size=BATCH_SIZE
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetResNet34MultiTask(
        n_classes_seg=3,
        n_classes_cls=n_classes_cls
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion_seg = nn.CrossEntropyLoss()
    criterion_cls = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0

        for batch in train_loader:
            x = batch["image"].to(device)
            y_seg = batch["mask"].to(device)
            y_cls = batch["label"].to(device)

            optimizer.zero_grad()

            seg_out, cls_out = model(x)

            loss = criterion_seg(seg_out, y_seg) + CLS_LOSS_WEIGHT * criterion_cls(cls_out, y_cls)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")


if __name__ == "__main__":
    main()

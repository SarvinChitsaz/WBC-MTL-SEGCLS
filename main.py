from configs.config import *
from data.prepare_data import prepare_dataset
from data.dataset import get_dataloaders
from data.transforms import get_train_transforms, get_test_transforms
from models.model import UNetResNet34MultiTask
import torch
import torch.nn as nn
import torch.optim as optim


def main():

    train_df, test_df, le = prepare_dataset(
        csv1_path, csv2_path, dataset1_dir, dataset2_dir
    )

    train_loader, test_loader = get_dataloaders(train_df, test_df, BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetResNet34MultiTask(
        n_classes_seg=3,
        n_classes_cls=len(le.classes_)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    criterion_seg = nn.CrossEntropyLoss()
    criterion_cls = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            seg_out, cls_out = model(images)

            loss = criterion_seg(seg_out, masks) + CLS_LOSS_WEIGHT * criterion_cls(cls_out, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")


if __name__ == "__main__":
    main()

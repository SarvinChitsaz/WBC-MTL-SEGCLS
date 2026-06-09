import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, seg_loss_fn, cls_loss_fn, device):
    model.train()

    total_loss = 0

    for batch in tqdm(loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        seg_out, cls_out = model(images)

        loss = seg_loss_fn(seg_out, masks) + 0.7 * cls_loss_fn(cls_out, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train(model, train_loader, optimizer, seg_loss, cls_loss, device, epochs):
    for epoch in range(epochs):
        loss = train_one_epoch(
            model, train_loader, optimizer,
            seg_loss, cls_loss, device
        )

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")

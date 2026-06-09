import torch
from data.dataset import get_dataloaders
from models.model import UNetResNet34MultiTask
from src.train import train
from src.eval import evaluate
from src.losses import get_losses
from configs.config import EPOCHS, LR, CLS_LOSS_WEIGHT


def main():
    print("HELLO FROM MAIN")
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1. data
    train_loader, test_loader, n_classes_cls = get_dataloaders()

    # 2. model
    model = UNetResNet34MultiTask(
        n_classes_seg=3,
        n_classes_cls=n_classes_cls
    ).to(device)

    # 3. loss + optimizer
    seg_loss, cls_loss = get_losses()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    # 4. training
    train(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        seg_loss=seg_loss,
        cls_loss=cls_loss,
        device=device,
        epochs=EPOCHS,
        cls_loss_weight=CLS_LOSS_WEIGHT
    )

    # 5. evaluation
    evaluate(model, test_loader, device)

    # 6. save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_classes_cls": n_classes_cls,
        "lr": LR,
        "epochs": EPOCHS
    }, "models/checkpoints/model.pth")

    print("Model saved successfully!")


if __name__ == "__main__":
    main()

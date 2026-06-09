import matplotlib.pyplot as plt
import numpy as np


SEG_COLORS = np.array([
    [0, 0, 0],
    [0, 200, 0],
    [255, 50, 50],
], dtype=np.uint8)


def mask_to_color(mask):
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for i, c in enumerate(SEG_COLORS):
        color[mask == i] = c

    return color


def show_results(images, masks, preds):
    images = images.cpu()
    masks = masks.cpu().numpy()
    preds = preds.argmax(dim=1).cpu().numpy()

    b = images.size(0)

    fig, axes = plt.subplots(b, 3, figsize=(8, 2*b))

    if b == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(b):
        img = images[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        axes[i, 0].imshow(img)
        axes[i, 1].imshow(mask_to_color(masks[i]))
        axes[i, 2].imshow(mask_to_color(preds[i]))

        axes[i, 0].set_title("Image")
        axes[i, 1].set_title("GT")
        axes[i, 2].set_title("Pred")

        for j in range(3):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()

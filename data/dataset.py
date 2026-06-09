import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from .label_utils import normalize_id, convert_mask


class WBCDataset(Dataset):
    def __init__(self, dataframe, transform=None, minority_classes=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.minority_classes = minority_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_id = normalize_id(row["image ID"])
        dir_ = row["dataset_dir"]

        img_path = os.path.join(dir_, img_id + ".bmp")
        mask_path = os.path.join(dir_, img_id + ".png")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask_raw = np.array(Image.open(mask_path).convert("L"))

        mask = convert_mask(mask_raw).astype("uint8")
        label = int(row["encoded_label"])

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()

        return {
            "image": image,
            "mask": mask,
            "label": label
        }

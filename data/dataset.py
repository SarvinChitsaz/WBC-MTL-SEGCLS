import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


from .label_utils import normalize_id, convert_mask


class WBCDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_id = normalize_id(row["image ID"])
        dir_ = row["dataset_dir"]

        img_path = os.path.join(dir_, img_id + ".bmp")
        mask_path = os.path.join(dir_, img_id + ".png")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        mask = convert_mask(mask).astype(np.uint8)
        label = int(row["encoded_label"])

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"].long()

        return {
            "image": image,
            "mask": mask,
            "label": label
        }

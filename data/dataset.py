import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from .label_utils import normalize_id, convert_mask


class WBCDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
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


def get_dataloaders(base_path, batch_size=8, test_size=0.2):

    dataset1_dir = os.path.join(base_path, "Dataset 1")
    dataset2_dir = os.path.join(base_path, "Dataset 2")

    csv1 = os.path.join(base_path, "Class Labels of Dataset 1.csv")
    csv2 = os.path.join(base_path, "Class Labels of Dataset 2.csv")

    df1 = pd.read_csv(csv1)
    df1["dataset_dir"] = dataset1_dir

    df2 = pd.read_csv(csv2)
    if "class" in df2.columns:
        df2 = df2.rename(columns={"class": "class label"})
    df2["dataset_dir"] = dataset2_dir

    df = pd.concat([df1, df2], ignore_index=True)

    # filter rare classes
    counts = df["class label"].value_counts()
    valid = counts[counts >= 20].index
    df = df[df["class label"].isin(valid)].reset_index(drop=True)

    # encoding
    le = LabelEncoder()
    df["encoded_label"] = le.fit_transform(df["class label"])

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df["class label"]
    )

    train_dataset = WBCDataset(train_df)
    test_dataset = WBCDataset(test_df)

    from torch.utils.data import DataLoader, WeightedRandomSampler

    class_counts = train_df["encoded_label"].value_counts().to_dict()
    weights = train_df["encoded_label"].apply(lambda x: 1.0 / class_counts[x])

    sampler = WeightedRandomSampler(
        weights=weights.values,
        num_samples=len(weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, len(le.classes_)

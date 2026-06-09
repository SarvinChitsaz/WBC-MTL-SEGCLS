import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .label_utils import normalize_id, convert_mask


class WBCDataset(Dataset):
    def __init__(self, df, transform=None, minority_classes=None):
        self.df = df.reset_index(drop=True)
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
        mask = np.array(Image.open(mask_path).convert("L"))

        mask = convert_mask(mask).astype(np.uint8)
        label = int(row["encoded_label"])

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"].long()

        return {"image": image, "mask": mask, "label": label}


def get_dataloaders(train_df, test_df, batch_size=8):

    class_counts = train_df["encoded_label"].value_counts().to_dict()
    weights = train_df["encoded_label"].apply(lambda x: 1.0 / class_counts[x])

    sampler = WeightedRandomSampler(
        weights=weights.values,
        num_samples=len(weights),
        replacement=True
    )

    train_ds = WBCDataset(train_df)
    test_ds = WBCDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader            image = aug["image"]
            mask = aug["mask"].long()

        return {"image": image, "mask": mask, "label": label}


def get_dataloaders(base_path, batch_size=8, test_size=0.2):

    dataset1_dir = os.path.join(base_path, "Dataset 1")
    dataset2_dir = os.path.join(base_path, "Dataset 2")

    csv1 = os.path.join(base_path, "Class Labels of Dataset 1.csv")
    csv2 = os.path.join(base_path, "Class Labels of Dataset 2.csv")

    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    if "class" in df2.columns:
        df2 = df2.rename(columns={"class": "class label"})

    df1["dataset_dir"] = dataset1_dir
    df2["dataset_dir"] = dataset2_dir

    df = pd.concat([df1, df2], ignore_index=True)

    le = LabelEncoder()
    df["encoded_label"] = le.fit_transform(df["class label"])

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df["class label"]
    )

    class_counts = train_df["encoded_label"].value_counts().to_dict()
    weights = train_df["encoded_label"].apply(lambda x: 1.0 / class_counts[x])

    sampler = WeightedRandomSampler(
        weights=weights.values,
        num_samples=len(weights),
        replacement=True
    )

    train_ds = WBCDataset(train_df)
    test_ds = WBCDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, len(le.classes_), le        mask_path = os.path.join(dir_, img_id + ".png")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask_raw = np.array(Image.open(mask_path).convert("L"))

        mask = convert_mask(mask_raw).astype(np.uint8)
        label = int(row["encoded_label"])

        if self.transform:
            if self.minority_classes is not None and label in self.minority_classes:
                augmented = minority_transform(image=image, mask=mask)
            else:
                augmented = self.transform(image=image, mask=mask)

            image = augmented["image"]
            mask = augmented["mask"].long()

        return {"image": image, "mask": mask, "label": label}


def get_dataloaders(base_path, batch_size=8, test_size=0.2):

    dataset1_dir = os.path.join(base_path, "Dataset 1")
    dataset2_dir = os.path.join(base_path, "Dataset 2")

    csv1_path = os.path.join(base_path, "Class Labels of Dataset 1.csv")
    csv2_path = os.path.join(base_path, "Class Labels of Dataset 2.csv")

    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    df1["dataset_dir"] = dataset1_dir
    df2["dataset_dir"] = dataset2_dir

    if "class" in df2.columns and "class label" not in df2.columns:
        df2 = df2.rename(columns={"class": "class label"})

    df = pd.concat([df1, df2], ignore_index=True)

    counts = df["class label"].value_counts()
    valid_classes = counts[counts >= 20].index
    df = df[df["class label"].isin(valid_classes)].reset_index(drop=True)

    le = LabelEncoder()
    df["encoded_label"] = le.fit_transform(df["class label"])
    n_classes_cls = len(le.classes_)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=50,
        stratify=df["class label"]
    )

    class_counts = train_df["encoded_label"].value_counts().to_dict()
    sample_weights = train_df["encoded_label"].apply(lambda x: 1.0 / class_counts[x])

    sampler = WeightedRandomSampler(
        weights=sample_weights.values,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_dataset = WBCDataset(train_df)
    test_dataset = WBCDataset(test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, n_classes_cls, le

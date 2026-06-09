import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import WBCDataset


def get_dataloaders(
    csv1_path,
    csv2_path,
    dataset1_dir,
    dataset2_dir,
    batch_size,
    test_size,
    train_transform,
    test_transform
):

    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    if "class" in df2.columns and "class label" not in df2.columns:
        df2 = df2.rename(columns={"class": "class label"})

    df1["dataset_dir"] = dataset1_dir
    df2["dataset_dir"] = dataset2_dir

    df = pd.concat([df1, df2], ignore_index=True)

    df = df.dropna()

    le = LabelEncoder()
    df["encoded_label"] = le.fit_transform(df["class label"])

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["class label"],
        random_state=42
    )

    train_ds = WBCDataset(train_df, train_transform)
    test_ds = WBCDataset(test_df, test_transform)

    class_counts = train_df["encoded_label"].value_counts().to_dict()
    weights = train_df["encoded_label"].apply(lambda x: 1.0 / class_counts[x])

    sampler = WeightedRandomSampler(
        weights=weights.values,
        num_samples=len(weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, len(le.classes_)

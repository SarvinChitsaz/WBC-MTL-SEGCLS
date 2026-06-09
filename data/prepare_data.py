import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .label_utils import normalize_id


def load_and_prepare(csv1_path, csv2_path, dataset1_dir, dataset2_dir):
    df1 = pd.read_csv(csv1_path)
    df1["dataset_dir"] = dataset1_dir

    df2 = pd.read_csv(csv2_path)
    if "class" in df2.columns and "class label" not in df2.columns:
        df2 = df2.rename(columns={"class": "class label"})
    df2["dataset_dir"] = dataset2_dir

    df = pd.concat([df1, df2], ignore_index=True)

    counts = df["class label"].value_counts()
    valid = counts[counts >= 20].index
    df = df[df["class label"].isin(valid)].reset_index(drop=True)

    def mask_exists(row):
        img_id = normalize_id(row["image ID"])
        dir_ = row["dataset_dir"]

        return (
            os.path.exists(os.path.join(dir_, img_id + ".bmp")) and
            os.path.exists(os.path.join(dir_, img_id + ".png"))
        )

    df = df[df.apply(mask_exists, axis=1)].reset_index(drop=True)

    le = LabelEncoder()
    df["encoded_label"] = le.fit_transform(df["class label"])

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=50,
        stratify=df["class label"]
    )

    return df, train_df, test_df, le

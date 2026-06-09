import pandas as pd

def inspect_dataset(df):
    print("Classes distribution:")
    print(df["class label"].value_counts())
    print("Total samples:", len(df))

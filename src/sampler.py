from torch.utils.data import WeightedRandomSampler

def get_sampler(train_df):
    class_counts = train_df["encoded_label"].value_counts().to_dict()

    sample_weights = train_df["encoded_label"].apply(
        lambda x: 1.0 / class_counts[x]
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights.values,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler

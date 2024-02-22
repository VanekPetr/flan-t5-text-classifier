import pandas as pd
from datasets import Dataset


def load_dataset() -> Dataset:
    """Load dataset."""
    dataset_ecommerce_pandas = pd.read_csv(
        "data/ecommerce_kaggle_dataset.csv", header=None, names=["label", "text"]
    )
    dataset_ecommerce_pandas["label"] = dataset_ecommerce_pandas["label"].astype(str)
    dataset_ecommerce_pandas["text"] = dataset_ecommerce_pandas["text"].astype(str)
    dataset = Dataset.from_pandas(dataset_ecommerce_pandas)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.2)

    return dataset

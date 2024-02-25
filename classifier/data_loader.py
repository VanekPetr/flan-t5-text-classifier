import os

import pandas as pd
from datasets import Dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

label2id = {"Books": 0, "Clothing & Accessories": 1, "Electronics": 2, "Household": 3}
id2label = {id: label for label, id in label2id.items()}


def load_dataset(model_type: str = "") -> Dataset:
    """Load dataset."""
    dataset_ecommerce_pandas = pd.read_csv(
        ROOT_DIR + "/data/ecommerce_kaggle_dataset.csv",
        header=None,
        names=["label", "text"],
    )

    dataset_ecommerce_pandas["label"] = dataset_ecommerce_pandas["label"].astype(str)
    if model_type == "AutoModelForSequenceClassification":
        # Convert labels to integers
        dataset_ecommerce_pandas["label"] = dataset_ecommerce_pandas["label"].map(
            label2id
        )

    dataset_ecommerce_pandas["text"] = dataset_ecommerce_pandas["text"].astype(str)
    dataset = Dataset.from_pandas(dataset_ecommerce_pandas)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.2)

    return dataset


if __name__ == "__main__":
    print(load_dataset())

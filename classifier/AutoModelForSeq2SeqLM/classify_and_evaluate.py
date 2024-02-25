import torch
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from classifier.data_loader import load_dataset

dataset = load_dataset()

# Load model and tokenizer from the hub
tokenizer = AutoTokenizer.from_pretrained(
    "VanekPetr/flan-t5-base-ecommerce-text-classification"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "VanekPetr/flan-t5-base-ecommerce-text-classification"
)
model.to("cuda" if torch.cuda.is_available() else "cpu")


def classify(texts_to_classify: str):
    """Classify a batch of texts using the model."""
    inputs = tokenizer(
        texts_to_classify,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=2,
            early_stopping=True,
        )

    predictions = [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]
    return predictions


def evaluate():
    """Evaluate the model on the test dataset."""
    predictions_list, labels_list = [], []

    batch_size = 16  # Adjust batch size based GPU capacity
    num_batches = len(dataset["test"]) // batch_size + (
        0 if len(dataset["test"]) % batch_size == 0 else 1
    )
    progress_bar = tqdm(total=num_batches, desc="Evaluating")

    for i in range(0, len(dataset["test"]), batch_size):
        batch_texts = dataset["test"]["text"][i : i + batch_size]
        batch_labels = dataset["test"]["label"][i : i + batch_size]

        batch_predictions = classify(batch_texts)

        predictions_list.extend(batch_predictions)
        labels_list.extend([str(label) for label in batch_labels])

        progress_bar.update(1)

    progress_bar.close()
    report = classification_report(labels_list, predictions_list)
    print(report)


if __name__ == "__main__":
    evaluate()

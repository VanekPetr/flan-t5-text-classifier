import nltk
import numpy as np
from huggingface_hub import HfFolder
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from classifier.data_loader import id2label, label2id, load_dataset

MODEL_ID = "google/flan-t5-small"
REPOSITORY_ID = f"{MODEL_ID.split('/')[1]}-ecommerce-text-classification"

config = AutoConfig.from_pretrained(
    MODEL_ID, num_labels=len(label2id), id2label=id2label, label2id=label2id
)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, config=config)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

training_args = TrainingArguments(
    num_train_epochs=2,
    output_dir=REPOSITORY_ID,
    logging_strategy="steps",
    logging_steps=100,
    report_to="tensorboard",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=False,  # Overflows with fp16
    learning_rate=3e-4,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=False,
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id=REPOSITORY_ID,
    hub_token=HfFolder.get_token(),
)


def tokenize_function(examples) -> dict:
    """Tokenize the text column in the dataset"""
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred) -> dict:
    """Compute metrics for evaluation"""
    logits, labels = eval_pred
    if isinstance(
        logits, tuple
    ):  # if the model also returns hidden_states or attentions
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def train() -> None:
    """
    Train the model and save it to the Hugging Face Hub.
    """
    dataset = load_dataset("AutoModelForSequenceClassification")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    nltk.download("punkt")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    # TRAIN
    trainer.train()

    # SAVE AND EVALUATE
    tokenizer.save_pretrained(REPOSITORY_ID)
    trainer.create_model_card()
    trainer.push_to_hub()
    print(trainer.evaluate())


if __name__ == "__main__":
    train()

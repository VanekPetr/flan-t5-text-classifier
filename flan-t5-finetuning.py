from typing import List, Tuple

import evaluate
import nltk
import numpy as np
from datasets import Dataset, concatenate_datasets
from huggingface_hub import HfFolder
from nltk.tokenize import sent_tokenize
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from data_loader import load_dataset

MODEL_ID = "google/flan-t5-base"
REPOSITORY_ID = f"{MODEL_ID.split('/')[1]}-ecommerce-text-classification"

# Load dataset
dataset = load_dataset()

# Load tokenizer of FLAN-t5
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Metric
metric = evaluate.load("f1")

# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x["text"], truncation=True),
    batched=True,
    remove_columns=["text", "label"],
)
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x["label"], truncation=True),
    batched=True,
    remove_columns=["text", "label"],
)
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"Max target length: {max_target_length}")

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=REPOSITORY_ID,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=False,  # Overflows with fp16
    learning_rate=3e-4,
    num_train_epochs=2,
    logging_dir=f"{REPOSITORY_ID}/logs",  # logging & evaluation strategies
    logging_strategy="epoch",
    evaluation_strategy="no",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=False,
    report_to="tensorboard",
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id=REPOSITORY_ID,
    hub_token=HfFolder.get_token(),
)


def preprocess_function(sample: Dataset, padding: str = "max_length") -> dict:
    """Preprocess the dataset."""

    # add prefix to the input for t5
    inputs = [item for item in sample["text"]]

    # tokenize inputs
    model_inputs = tokenizer(
        inputs, max_length=max_source_length, padding=padding, truncation=True
    )

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=sample["label"],
        max_length=max_target_length,
        padding=padding,
        truncation=True,
    )

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(la if la != tokenizer.pad_token_id else -100) for la in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(
    preds: List[str], labels: List[str]
) -> Tuple[List[str], List[str]]:
    """helper function to postprocess text"""
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, average="macro"
    )
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result


def train() -> None:
    """Train the model."""

    tokenized_dataset = dataset.map(
        preprocess_function, batched=True, remove_columns=["text", "label"]
    )
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    nltk.download("punkt")

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )

    # TRAIN
    trainer.train()

    # SAVE
    tokenizer.save_pretrained(REPOSITORY_ID)
    trainer.create_model_card()
    trainer.push_to_hub()


if __name__ == "__main__":
    train()

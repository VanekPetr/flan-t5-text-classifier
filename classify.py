import torch
from tqdm.auto import tqdm
from data_loader import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import classification_report

dataset = load_dataset()

# Load model and tokenizer from the hub
tokenizer = AutoTokenizer.from_pretrained("VanekPetr/flan-t5-base-ecommerce-text-classification")
model = AutoModelForSeq2SeqLM.from_pretrained("VanekPetr/flan-t5-base-ecommerce-text-classification")
model.to('cuda') if torch.cuda.is_available() else model.to('cpu')


def classify(text_to_classify: str) -> str:
    """Classify a text using the model."""
    inputs = tokenizer.encode_plus(text_to_classify, padding='max_length', max_length=512, return_tensors='pt')
    inputs = inputs.to('cuda') if torch.cuda.is_available() else inputs.to('cpu')
    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=150, num_beams=4, early_stopping=True)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction


if __name__ == '__main__':
    predictions_list, labels_list = [], []

    samples_number = len(dataset['test'])
    progress_bar = tqdm(range(samples_number))

    for i in range(samples_number):
        text = dataset['test']['text'][i]
        predictions_list.append(classify(text))
        labels_list.append(str(dataset['test']['label'][i]))

        progress_bar.update(1)

    report = classification_report(labels_list, predictions_list)
    print(report)

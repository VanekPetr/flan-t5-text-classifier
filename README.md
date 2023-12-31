# flan-t5 Fine-tuning for Text Classification

This GitHub project is aimed at fine-tuning the flan t5 model for a text classification task using an E-commerce 
text dataset for 4 categories - "Electronics", "Household", "Books" and "Clothing & Accessories".

## Results
<p>
  <img width="100%" src="data/evaluation.png"></a>
</p>

## Dataset

The dataset is a classification-based E-commerce text dataset, which almost covers 80% of any E-commerce website. The dataset consists of product and description data for 4 categories. The dataset can be found [here](https://doi.org/10.5281/zenodo.3355823).

## Project Features

The project employs the tokenizer of flan-t5 by Hugging Face, which helps in splitting the input text into a format that is understandable by the model.

An evaluation function has been implemented for post-processing the labels and predictions, which will also handle sequence length adjustments.

The project uses a Seq2SeqTrainer for training the model. It also includes a helper function to preprocess the dataset.

## Usage

To leverage the project you need to run the `flan-t5-finetuning.py` script which will trigger the training of the model.

The 'train' function fine-tunes the flan-t5 model, trains it with the dataset, outputs the metrics, creates a model card and pushes the model to Hugging Face model hub.

The preprocess function tokenizes the inputs, and also handles tokenization of the target labels. The compute_metrics function evaluates the model performance based on the F1 metric.

## Requirements

You need to install the required Python packages mentioned in the `requirements.txt` file.

Please refer to the source code for more detailed information about the implementation.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/VanekPetr/flan-t5-text-classifier/tags). 

## License

This repository is licensed under [MIT](LICENSE) (c) 2023 GitHub, Inc.
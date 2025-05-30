# Text Mining - Performing High-Level NLP Tasks
This project demonstrates the application of three advanced NLP techniques:
- Named Entitiy Recognition and Classification (NERC)
- Sentiment Analysis
- Topic Classification

The goal is to implement and compare different models across these tasks using different approaches.

**NOTE: The instructions for usage of the models are provided in the Jupyter Notebooks.**

## 📁 Project Structure
- [`nerc_crf.ipynb`](nerc_crf.ipynb) - CRF-based NER
- [`nerc_bert.ipynb`](nerc_bert.ipynb) - BERT fine-tuning for NER
- [`data`](data/) - Folder containing the provided and external datasets
- [`utils.py`](utils.py) - Utility function for preprocessing and formatting

## Named Entity Recognition and Classification (NERC)
NERC is a NLP technique that recognises and labels nammed entities (e.g, locations, persons, organisations) in text.

### Dataset
The test dataset was provided by the university, [`NER-test.tsv`](data/test_data/NER-test.tsv). The training dataset can be found on HuggingFace:

- [`CoNLL-2003 Dataset`](https://huggingface.co/datasets/eriktks/conll2003)

Since it was quite difficult to align with the provided test data, we applied spaCy's NER tags for alignment.

## NERC Model 1 - CRF
The CRF was implemented using sklearn-crfsuite, and was trained on tokenised text with IOB-labels. - [`nerc_crf.ipynb`](nerc_crf.ipynb)

## NERC Model 2 - BERT
BERT was fine-tuned (bert-base-uncased) using HuggingFace Transformers on token-level annotations. The tokenisation was aligned using word_ids(), and training was done using the Trainer API. - [`nerc_bert.ipynb`](nerc_bert.ipynb)

## Sentiment Analysis
...

## Topic Analysis
...

## Setup and installation
To start using the repo, the requirements should be installed first:

```python
pip install -r requirements.txt
```

For the setup, we recommend using a conda environment:

```python
conda create -n nlp_env python=3.12
conda activate nlp_env
```




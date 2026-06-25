# 🧠 Multilingual NLP Text Classifier (BERT / RoBERTa / DistilBERT)

A production-ready **multilingual text classification** system using Hugging Face Transformers.
Fine-tune and compare BERT, RoBERTa, and DistilBERT on any text classification task.

*Based on M.Sc. thesis research in multilingual transformer-based document classification.*

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow?style=flat-square)

## Quick Start
```bash
git clone https://github.com/OsamaBinShah1/multilingual-nlp-classifier.git
cd multilingual-nlp-classifier
pip install -r requirements.txt
python train.py --data data/sample.csv --model bert-base-multilingual-cased --epochs 3
python predict.py --text "Das ist ein sehr guter Artikel" --model_path ./saved_model
```

## Supported Models
| Model | Languages | Best For |
|-------|-----------|----------|
| `bert-base-multilingual-cased` | 104 | Cross-lingual tasks |
| `roberta-base` | English | English accuracy |
| `distilbert-base-multilingual-cased` | 104 | Speed + efficiency |
| `xlm-roberta-base` | 100 | Best multilingual |

## Author
**Muhammad Osama Bin Shah** — AI Engineer, Frankfurt, Germany
M.Sc. thesis: Multilingual transformer-based document classification
[LinkedIn](https://www.linkedin.com/in/muhammad-osama-bin-shah/)

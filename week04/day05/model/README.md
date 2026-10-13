---
language:
- en
library_name: transformers
pipeline_tag: text-classification
base_model: distilbert/distilbert-base-uncased
tags:
- customer-support
- text-classification
- learning-project
---

# Customer Support DistilBERT

A DistilBERT model fine-tuned to classify English customer support messages into four categories:

| ID | Label | Description |
|---|---|---|
| 0 | payment | Payment failures, payment methods, and transaction issues |
| 1 | product_issue | Product damage, missing parts, and quality or functionality issues |
| 2 | refund | Refund requests, refund status, and refund amount issues |
| 3 | shipping | Delivery, tracking, and shipping address issues |

## Intended Use

This model is a personal learning project demonstrating data preparation, fine-tuning, evaluation, and error analysis. It is not ready for production customer support routing.

## Training

- Base model: `distilbert/distilbert-base-uncased`
- Dataset: 79 cleaned and deduplicated customer support examples
- Split: 47 training, 16 validation, and 16 test examples
- Learning rate: `2e-5`
- Training batch size: `4`
- Epochs: `3`
- Seed: `42`
- Selected checkpoint: `checkpoint-24`, based on validation accuracy

## Validation Results

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| payment | 0.0000 | 0.0000 | 0.0000 | 4 |
| product_issue | 0.8000 | 1.0000 | 0.8889 | 4 |
| refund | 0.3750 | 0.7500 | 0.5000 | 4 |
| shipping | 1.0000 | 0.7500 | 0.8571 | 4 |

- Validation accuracy: **0.6250**
- Validation macro F1: **0.5615**
- The test split has not been evaluated.

## Limitations

The dataset and validation set are small. All four validation payment examples were misclassified as refund. Six validation errors have been analyzed. Results do not establish reliable performance on real customer support messages.

## Usage

Authenticate with an account that has access to this private repository:

```bash
hf auth login
```

Load the model and tokenizer:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo_id = "CHENXI0312/customer-support-distilbert"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForSequenceClassification.from_pretrained(repo_id)

inputs = tokenizer(
    "Where is my package?",
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

model.eval()

with torch.no_grad():
    logits = model(**inputs).logits

predicted_id = logits.argmax(dim=-1).item()
print(model.config.id2label[predicted_id])
```
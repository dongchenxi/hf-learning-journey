from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch


model_name = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name
)

text = "I love this movie!"

inputs = tokenizer(
    text,
    return_tensors="pt",
)

with torch.no_grad():
    outputs = model(**inputs)

print("Logits:")
print(outputs.logits)

print("\nLogits shape:")
print(outputs.logits.shape)

print("\nid2label:")
print(model.config.id2label)
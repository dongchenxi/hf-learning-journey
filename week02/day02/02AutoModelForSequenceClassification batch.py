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

texts = [
    "I love this movie!",
    "This movie is terrible.",
    "The movie is okay.",
    "I hate this product.",
]
inputs = tokenizer(
    texts,
    return_tensors="pt",
)

with torch.no_grad():
    outputs = model(**inputs)

probability=torch.softmax(outputs.logits,dim=-1)
print(probability)
class_ids=torch.argmax(probability,dim=-1)
print(class_ids)
for text, class_id in zip(texts, class_ids):
    label = model.config.id2label[class_id.item()]
    print(text, "->", label)


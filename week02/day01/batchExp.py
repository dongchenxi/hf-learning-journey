from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
model_name="distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
texts = [
    "I love this movie!",
    "This movie is terrible.",
    "The movie is okay.",
    "I hate this product.",
]
inputs = tokenizer(
    texts,
    return_tensors="pt"
)
outputs = model(**inputs)
print(outputs.logits)
probabilities=torch.softmax(outputs.logits,dim=-1)
print(probabilities)
predicted_class_ids=torch.argmax(probabilities,dim=-1)
for class_id in predicted_class_ids:
    label = model.config.id2label[class_id.item()]
    print(label)


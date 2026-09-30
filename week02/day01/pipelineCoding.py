from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(
    "I love Hugging Face",
    return_tensors="pt"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
outputs = model(**inputs)
prob = torch.softmax(
    outputs.logits,
    dim=-1
)
prediction = torch.argmax(prob,dim=-1)
print(prediction)
print(model.config.id2label)
import token

from transformers import AutoTokenizer, AutoModel
import torch


model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
texts = [
    "Hello!",
    "I love machine learning.",
]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
print(inputs["input_ids"].shape)
# torch.Size([2, 7]) 2条text，分别切割成7个token

# print(outputs.last_hidden_state.shape)
# torch.Size([2, 7, 768]) 2条text，分别切割成7个token,Token 在最后一层得到的向量表示
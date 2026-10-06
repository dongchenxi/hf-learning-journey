from transformers import AutoTokenizer, AutoModel
import torch


model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "I love Hugging Face!"

inputs = tokenizer(
    text,
    return_tensors="pt",
)

with torch.no_grad():
    outputs = model(**inputs)

print("Input IDs:")
print(inputs["input_ids"])

print("\nInput shape:")
print(inputs["input_ids"].shape)

print("\nLast hidden state:")
print(outputs.last_hidden_state)

print("\nLast hidden state shape:")
print(outputs.last_hidden_state.shape)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch


model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name
)

text = "I love machine learning"

inputs = tokenizer(
    text,
    return_tensors="pt",
)

with torch.no_grad():
    outputs = model(**inputs)

print("\nInput shape:")
print(inputs["input_ids"].shape)

print("\nLogits shape:")
print(outputs.logits.shape)

next_token_logits = outputs.logits[:, -1, :]
print("\nNext token logits shape:")
print(next_token_logits.shape)

next_token_ids = torch.argmax(next_token_logits, dim=-1)
print("\nNext token ids:")
print(next_token_ids)
next_token=tokenizer.decode(next_token_ids)
print("\nNext token:")
print(next_token)
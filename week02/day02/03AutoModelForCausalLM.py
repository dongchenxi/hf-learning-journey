from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = AutoModelForCausalLM.from_pretrained("gpt2")

inputs = tokenizer(
    "Hugging Face is",
    return_tensors="pt"
)

outputs = model.generate(
    **inputs,
    max_new_tokens=30
)

print(
    tokenizer.decode(outputs[0])
)
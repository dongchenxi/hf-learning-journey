import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = "Qwen/Qwen2.5-0.5B"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"


print("1. Loading tokenizer...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("2. Tokenizer loaded!", flush=True)

print("3. Loading model...", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

print("4. Model loaded!", flush=True)

prompt = "What is Python?"

print("5. Tokenizing...", flush=True)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to(model.device)

print("6. Generating...", flush=True)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

print("7. Generation finished!", flush=True)

generated_text = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)

print("Generated:", generated_text)
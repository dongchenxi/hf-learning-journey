from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "Artificial intelligence will"

inputs = tokenizer(
    prompt,
    return_tensors="pt",
)

values = [10, 30, 60]

for value in values:
    print("=" * 80)
    print("max_new_tokens:", value)

    outputs = model.generate(
        **inputs,
        max_new_tokens=value,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    print(text)
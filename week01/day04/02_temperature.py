from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)

model_id = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "Artificial intelligence will"

inputs = tokenizer(
    prompt,
    return_tensors="pt",
)

temperatures = [0.2, 0.7, 1.2]

for temperature in temperatures:
    print("=" * 80)
    print("temperature:", temperature)

    set_seed(42)

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    print(text)
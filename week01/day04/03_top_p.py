from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)

model_id = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "Machine learning is"

inputs = tokenizer(
    prompt,
    return_tensors="pt",
)

top_p_values = [0.5, 0.8, 0.95]

for top_p in top_p_values:
    print("=" * 80)
    print("top_p:", top_p)

    set_seed(42)

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=1.0,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    print(text)
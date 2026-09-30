from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)

model_id = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "The future of software engineering is"

inputs = tokenizer(
    prompt,
    return_tensors="pt",
)

top_k_values = [5, 20, 50]

for top_k in top_k_values:
    print("=" * 80)
    print("top_k:", top_k)

    set_seed(42)

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=1.0,
        top_k=top_k,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    print(text)
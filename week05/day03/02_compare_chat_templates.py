from transformers import AutoTokenizer

model_names = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "HuggingFaceH4/zephyr-7b-beta"
]

messages = [
    {
        "role": "system",
        "content": "You are a Python teacher."
    },
    {
        "role": "user",
        "content": "What is Python?"
    }
]

for model_name in model_names:
    print(f"\nModel: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(formatted_text)
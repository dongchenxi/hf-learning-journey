from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

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

formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("Formatted text:")
print(formatted_text)
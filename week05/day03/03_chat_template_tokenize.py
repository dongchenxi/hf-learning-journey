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

# 1. 返回格式化文本
formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 2. 返回 Token IDs
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True
)

# 3. 将 Token IDs 解码为文本
decoded_text = tokenizer.decode(input_ids)

print("Formatted text:")
print(formatted_text)

print("\nToken IDs:")
print(input_ids)

print("\nDecoded text:")
print(decoded_text)

print("\nSame text:", formatted_text == decoded_text)
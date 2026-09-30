from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

encoding = tokenizer(
    "I love Hugging Face"
)
print(encoding)
print(encoding["input_ids"])
print(tokenizer.convert_ids_to_tokens(
    encoding["input_ids"]
))
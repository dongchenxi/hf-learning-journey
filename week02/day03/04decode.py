from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

encoding = tokenizer(
    "I love Hugging Face"
)
print(encoding)
print(encoding["input_ids"])
ids=encoding["input_ids"]
print(
    tokenizer.decode(ids, skip_special_tokens=True)
)
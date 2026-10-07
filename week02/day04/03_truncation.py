from transformers import AutoTokenizer


model_name = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

text = (
    "Machine learning and natural language processing "
    "are very interesting fields of artificial intelligence."
)


without_truncation = tokenizer(
    text
)

with_truncation = tokenizer(
    text,
    truncation=True,
    max_length=10,
)


print("Without truncation:")
print(without_truncation["input_ids"])
print(
    "Length:",
    len(without_truncation["input_ids"]),
)


print("\nWith truncation:")
print(with_truncation["input_ids"])
print(
    "Length:",
    len(with_truncation["input_ids"]),
)
print(
    tokenizer.decode(
        without_truncation["input_ids"]
    )
)

print(
    tokenizer.decode(
        with_truncation["input_ids"]
    )
)
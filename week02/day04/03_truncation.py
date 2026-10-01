from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

texts = [
    "I love Hugging Face",
    "Transformers are amazing",
    "Deep learning"
]
long_text = "This is a very long sentence ..." * 100
encoding = tokenizer(
    long_text,
    truncation=True,
    max_length=16
)

print(len(encoding["input_ids"]))
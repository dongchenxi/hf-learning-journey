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
    texts,
    padding=True,
    return_tensors="pt"
)

print(encoding["attention_mask"])

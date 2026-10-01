from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

texts = [
    "I love Hugging Face",
    "Transformers are amazing",
    "Deep learning"
]

encoding = tokenizer(texts)

print(encoding)
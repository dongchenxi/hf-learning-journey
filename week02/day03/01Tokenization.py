from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

tokens = tokenizer.tokenize(
    "I love Hugging Face"
)

print(tokens)
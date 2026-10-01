from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
model_name="distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
texts = [
    "I love this movie!",
    "This movie is terrible.",
    "The movie is okay.",
    "I hate this product.",
]
for text in texts:
    tokens = tokenizer.tokenize(text)
    token_id=tokenizer.convert_tokens_to_ids(tokens)
    print(tokens)
    print(token_id)
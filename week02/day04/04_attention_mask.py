from transformers import AutoTokenizer


model_name = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

texts = [
    "Hello!",
    "I really love machine learning!",
]

inputs = tokenizer(
    texts,
    padding=True,
    return_tensors="pt",
)

print("Input IDs:")
print(inputs["input_ids"])

print("\nAttention Mask:")
print(inputs["attention_mask"])

print("\nInput IDs shape:")
print(inputs["input_ids"].shape)

print("\nAttention Mask shape:")
print(inputs["attention_mask"].shape)
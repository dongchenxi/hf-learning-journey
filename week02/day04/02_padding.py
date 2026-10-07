from transformers import AutoTokenizer


model_name = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

texts = [
    "Hello!",
    "I really love machine learning!",
    "Transformers are amazing.",
]

inputs = tokenizer(
    texts,
    padding=True,
    return_tensors="pt",
)

print("Input IDs:")
print(inputs["input_ids"])

print("\nShape:")
print(inputs["input_ids"].shape)

for input_ids in inputs["input_ids"]:
    tokens = tokenizer.convert_ids_to_tokens(
        input_ids
    )

    print(tokens)
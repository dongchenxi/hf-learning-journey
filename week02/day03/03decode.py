from transformers import AutoTokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)

texts = [
    "I love this movie!",
    "Machine learning is amazing.",
    "unbelievably",
]


for text in texts:
    inputs = tokenizer(text)

    input_ids = inputs["input_ids"]

    tokens = tokenizer.convert_ids_to_tokens(
        input_ids
    )

    decoded = tokenizer.decode(
        input_ids
    )

    decoded_clean = tokenizer.decode(
        input_ids,
        skip_special_tokens=True,
    )

    print("Original:")
    print(text)

    print("Input IDs:")
    print(input_ids)

    print("Tokens:")
    print(tokens)

    print("Decoded:")
    print(decoded)

    print("Decoded clean:")
    print(decoded_clean)

    print("-" * 50)
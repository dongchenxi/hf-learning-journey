from transformers import AutoTokenizer


model_name = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

texts = [
    "Hello!",
    "I love machine learning.",
    (
        "Natural language processing is a very "
        "interesting area of artificial intelligence."
    ),
]

inputs = tokenizer(
    texts,
    padding=True,#发生了padding
    truncation=True,#发生了truncation
    max_length=12,
    return_tensors="pt",
)


print("Input IDs:")
print(inputs["input_ids"])

print("\nAttention Mask:")
print(inputs["attention_mask"])

print("\nInput IDs shape:")
print(inputs["input_ids"].shape)


for i in range(len(texts)):
    print("\nSample:", i)

    input_ids = inputs["input_ids"][i]
    attention_mask = inputs["attention_mask"][i]

    tokens = tokenizer.convert_ids_to_tokens(
        input_ids
    )

    print("Original:")
    print(texts[i])

    print("Tokens:")
    print(tokens)

    print("Input IDs:")
    print(input_ids)

    print("Attention Mask:")
    print(attention_mask)

    print("Decoded:")
    print(
        tokenizer.decode(
            input_ids,
            skip_special_tokens=True,
        )
    )
from transformers import AutoTokenizer


model_name = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


print("Special tokens map:")
print(tokenizer.special_tokens_map)


print("\nAll special tokens:")
print(tokenizer.all_special_tokens)


print("\nAll special token IDs:")
print(tokenizer.all_special_ids)


print("\nToken -> ID:")

for token in tokenizer.all_special_tokens:
    token_id = tokenizer.convert_tokens_to_ids(
        token
    )

    print(token, "->", token_id)
from transformers import AutoTokenizer


model_name = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

text = "I love this movie!"


# 1. Raw text
print("Raw text:")
print(text)


# 2. Tokens
tokens = tokenizer.tokenize(text)

print("\nTokens:")
print(tokens)


# 3. Token IDs without automatically adding special tokens
token_ids = tokenizer.convert_tokens_to_ids(
    tokens
)

print("\nToken IDs:")
print(token_ids)


# 4. Complete tokenizer output
inputs = tokenizer(text)

print("\nTokenizer output:")
print(inputs)


# 5. input_ids
input_ids = inputs["input_ids"]

print("\nInput IDs:")
print(input_ids)


# 6. Convert input IDs back to tokens
input_tokens = tokenizer.convert_ids_to_tokens(
    input_ids
)

print("\nInput tokens:")
print(input_tokens)


# 7. Decode
decoded = tokenizer.decode(input_ids)

print("\nDecoded:")
print(decoded)


# 8. Decode without special tokens
decoded_clean = tokenizer.decode(
    input_ids,
    skip_special_tokens=True,
)

print("\nDecoded without special tokens:")
print(decoded_clean)
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

bert = AutoModel.from_pretrained(
    "google-bert/bert-base-uncased"
)

gpt2 = AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2"
)

t5 = AutoModelForSeq2SeqLM.from_pretrained(
    "google-t5/t5-small"
)

print("BERT:")
print(type(bert))

print("\nGPT-2:")
print(type(gpt2))

print("\nT5:")
print(type(t5))
from datasets import load_dataset
from transformers import AutoTokenizer


model_name = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

dataset = load_dataset(
    "csv",
    data_files="data/customer_support.csv",
    split="train",
)


def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
    )


tokenized_dataset = dataset.map(
    tokenize_function,
batched = True

)

print(tokenized_dataset)
print(tokenized_dataset[0])
# print(dataset.features)
# print(tokenized_dataset.features)
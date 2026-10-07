from datasets import load_dataset,DatasetDict
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
                       split="train"
                       )
#清洗
def has_valid_text(example):
    return (
        example["text"] is not None
        and example["text"].strip() != ""
    )
clean_dataset = dataset.filter(
    has_valid_text
)
first_split=clean_dataset.train_test_split(
    test_size=0.2,
    seed=42
)
validation_test_split=first_split["test"].train_test_split(
    test_size=0.5,
    seed=42
)
#原始，清洗，划分后的
dataset_dict=DatasetDict({"train": first_split["train"],
    "validation": validation_test_split["train"],
    "test": validation_test_split["test"],}

)
#tokenizer后的dataset
def tokenize_function(batch):
    return tokenizer(batch["text"], truncation=True)

tokenized_dataset=dataset_dict.map(tokenize_function,batched=True)
print(tokenized_dataset)

print(
    tokenized_dataset["train"].features
)

print(
    tokenized_dataset["train"][0]
)

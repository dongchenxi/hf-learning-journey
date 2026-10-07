import csv

from datasets import load_dataset

dataset = load_dataset(
    "csv",
    data_files="data/customer_support.csv",
    split="train",
)
print("=== Original Dataset ===")
print(dataset)

# map()
def add_text_length(example):
    example["text_length"] = len(example["text"])
    return example


mapped_dataset = dataset.map(
    add_text_length
)

print("\n=== Mapped Dataset ===")
print(mapped_dataset)
print(mapped_dataset[0])

# filter()
def is_refund(example):
    return example["label"] == "refund"


refund_dataset = mapped_dataset.filter(
    is_refund
)

print("\n=== Filtered Dataset ===")
print(refund_dataset)

# select()
selected_dataset = mapped_dataset.select(
    range(3)
)

print("\n=== Selected Dataset ===")
print(selected_dataset)

# shuffle()
shuffled_dataset = mapped_dataset.shuffle(
    seed=42
)

print("\n=== Shuffled Dataset ===")
print(shuffled_dataset[:5])

# random sample
sample_dataset = (
    mapped_dataset
    .shuffle(seed=42)
    .select(range(5))
)

print("\n=== Random Sample ===")

for example in sample_dataset:
    print(example)
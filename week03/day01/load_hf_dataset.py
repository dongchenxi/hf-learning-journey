from datasets import load_dataset
dataset = load_dataset(
    "cornell-movie-review-data/rotten_tomatoes"
)

train_dataset = dataset["train"]
label_feature = train_dataset.features["label"]
for i in range(5):
    example = train_dataset[i]
    label_id=example["label"]
    label_name=label_feature.int2str(label_id)
    print(f"Example {i}")
    print(f"Text: {example['text']}")
    print(f"Label ID: {label_id}")
    print(f"Label Name: {label_name}")
    print("-" * 50)
from datasets import load_dataset
dataset = load_dataset(
    "cornell-movie-review-data/rotten_tomatoes"
)
print("=== Dataset ===")
print(dataset)

print("\n=== Dataset Type ===")
print(type(dataset))

print("\n=== Splits ===")
print(dataset.keys())

train_dataset = dataset["train"]
print("\n=== Train Dataset ===")
print(train_dataset["text"])

# print("\n=== Train Dataset Type ===")
# print(type(train_dataset))
#
# print("\n=== Number of Rows ===")
# print(train_dataset.num_rows)
#
# print("\n=== Columns ===")
# print(train_dataset.column_names)
#
# print("\n=== Features ===")
# print(train_dataset.features)
#
# print("\n=== First Example ===")
# print(train_dataset[0])
#
# label_feature = train_dataset.features["label"]
#
# for i in range(5):
#     example = train_dataset[i]
#
#     label_id = example["label"]
#     label_name = label_feature.int2str(label_id)
#
#     print(f"\nExample {i}")
#     print(f"Text: {example['text']}")
#     print(f"Label ID: {label_id}")
#     print(f"Label Name: {label_name}")
print(type(dataset))
print(type(train_dataset))
print(type(train_dataset[0]))
print(type(train_dataset[0]["text"]))
print(type(train_dataset[0]["label"]))
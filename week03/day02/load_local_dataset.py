from datasets import load_dataset


dataset = load_dataset(
    "csv",
    data_files="customer_support.csv",
)

print("=== Original DatasetDict ===")
print(dataset)

full_dataset = dataset["train"]

print("\n=== Dataset ===")
print(full_dataset)

print("\n=== Columns ===")
print(full_dataset.column_names)

print("\n=== Features ===")
print(full_dataset.features)

print("\n=== Number of Rows ===")
print(full_dataset.num_rows)

print("\n=== First Example ===")
print(full_dataset[0])


split_dataset = full_dataset.train_test_split(
    test_size=0.2,
    seed=43,
)

print("\n=== Split Dataset ===")
print(split_dataset)

print("\n=== Train Dataset ===")
print(split_dataset["train"])

print("\n=== Test Dataset ===")
print(split_dataset["test"][:])

print(
    "\nTrain rows:",
    split_dataset["train"].num_rows,
)

print(
    "Test rows:",
    split_dataset["test"].num_rows,
)
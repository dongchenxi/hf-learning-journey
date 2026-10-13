from pathlib import Path

import numpy as np
from datasets import DatasetDict, load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / 'week03'
    / 'day05'
    / 'data'
    / 'customer_support.csv'
)

TOKENIZER_CHECKPOINT = 'distilbert/distilbert-base-uncased'

MODEL_CHECKPOINT = (
    PROJECT_ROOT
    / 'week04'
    / 'day02'
    / 'outputs'
    / 'checkpoint-24'
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / 'week04'
    / 'day03'
    / 'outputs'
)

VALID_LABELS = {
    'shipping',
    'refund',
    'payment',
    'product_issue',
}

SEED = 42

set_seed(SEED)

if not MODEL_CHECKPOINT.is_dir():
    raise FileNotFoundError(
        f'Checkpoint not found: {MODEL_CHECKPOINT}'
    )


def has_valid_example(example):
    text = example['text']

    return (
        text is not None
        and text.strip() != ''
        and example['label'] in VALID_LABELS
    )


def remove_duplicate_texts(dataset):
    seen_texts = set()
    keep_indices = []

    for index, text in enumerate(dataset['text']):
        if text not in seen_texts:
            seen_texts.add(text)
            keep_indices.append(index)

    return dataset.select(keep_indices)


# 1. Load, clean, and deduplicate.
dataset = load_dataset(
    'csv',
    data_files=str(DATA_PATH),
    split='train',
)

clean_dataset = dataset.filter(has_valid_example)
deduplicated_dataset = remove_duplicate_texts(clean_dataset)

encoded_dataset = deduplicated_dataset.class_encode_column(
    'label'
)


# 2. Reproduce the training-time splits.
first_split = encoded_dataset.train_test_split(
    test_size=0.4,
    seed=SEED,
    stratify_by_column='label',
)

validation_test_split = first_split['test'].train_test_split(
    test_size=0.5,
    seed=SEED,
    stratify_by_column='label',
)

dataset_dict = DatasetDict({
    'train': first_split['train'],
    'validation': validation_test_split['train'],
    'test': validation_test_split['test'],
})


# 3. Tokenize.
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_CHECKPOINT
)


def tokenize_function(batch):
    return tokenizer(
        batch['text'],
        truncation=True,
        max_length=128,
    )


tokenized_dataset = dataset_dict.map(
    tokenize_function,
    batched=True,
)

columns_to_remove = [
    column
    for column in ('text', 'token_type_ids')
    if column in tokenized_dataset['train'].column_names
]

training_dataset = tokenized_dataset.remove_columns(
    columns_to_remove
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
)


# 4. Load the fine-tuned model and verify label IDs.
label_names = training_dataset['validation'].features['label'].names
label_ids = list(range(len(label_names)))

id2label = {
    index: name
    for index, name in enumerate(label_names)
}

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_CHECKPOINT)
)

if model.config.id2label != id2label:
    raise ValueError(
        'Dataset label mapping does not match the checkpoint.'
    )


# 5. Create an evaluation Trainer.
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_eval_batch_size=4,
    dataloader_pin_memory=False,
    report_to='none',
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    data_collator=data_collator,
)


# 6. Predict on validation.
prediction_output = trainer.predict(
    training_dataset['validation']
)

logits = prediction_output.predictions
true_ids = prediction_output.label_ids

predicted_ids = np.argmax(logits, axis=-1)


# 7. Calculate metrics.
accuracy = accuracy_score(
    true_ids,
    predicted_ids,
)

macro_f1 = f1_score(
    true_ids,
    predicted_ids,
    labels=label_ids,
    average='macro',
    zero_division=0,
)

report = classification_report(
    true_ids,
    predicted_ids,
    labels=label_ids,
    target_names=label_names,
    digits=4,
    zero_division=0,
)


# 8. Print the evaluation results.
print('\n--- Validation Evaluation ---')
print(f'Accuracy: {accuracy:.4f}')
print(f'Macro F1: {macro_f1:.4f}')
print(report)
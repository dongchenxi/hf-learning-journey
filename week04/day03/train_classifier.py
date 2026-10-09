from pathlib import Path

from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
    TrainingArguments,
    Trainer
)
import numpy as np
import torch

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

if not MODEL_CHECKPOINT.is_dir():
    raise FileNotFoundError(
        f'Checkpoint not found: {MODEL_CHECKPOINT}'
    )
VALID_LABELS = {
    'shipping',
    'refund',
    'payment',
    'product_issue',
}

set_seed(42)


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


# Load, clean, and deduplicate.
dataset = load_dataset(
    'csv',
    data_files=str(DATA_PATH),
    split='train',
)

clean_dataset = dataset.filter(has_valid_example)
deduplicated_dataset = remove_duplicate_texts(clean_dataset)
encoded_dataset = deduplicated_dataset.class_encode_column('label')


# Create train / validation / test splits.
first_split = encoded_dataset.train_test_split(
    test_size=0.4,
    seed=42,
    stratify_by_column='label',
)

validation_test_split = first_split['test'].train_test_split(
    test_size=0.5,
    seed=42,
    stratify_by_column='label',
)

dataset_dict = DatasetDict({
    'train': first_split['train'],
    'validation': validation_test_split['train'],
    'test': validation_test_split['test'],
})


# Tokenize the datasets.
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CHECKPOINT)


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


# Build label mappings and load the classification model.
label_names = training_dataset['train'].features['label'].names

id2label = {
    index: name
    for index, name in enumerate(label_names)
}

label2id = {
    name: index
    for index, name in id2label.items()
}

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_CHECKPOINT)
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicted_ids = np.argmax(logits, axis=-1)
    accuracy = (predicted_ids == labels).mean()
    return {
        'accuracy': float(accuracy),
    }

OUTPUT_DIR = (
    PROJECT_ROOT
    / 'week04'
    / 'day03'
    / 'outputs'
)


training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_eval_batch_size=4,
    dataloader_pin_memory=False,
    report_to='none',
    seed=42,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_dataset['train'],
    eval_dataset=training_dataset['validation'],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

prediction_output = trainer.predict(
    training_dataset['validation']
)

logits = prediction_output.predictions
true_ids = prediction_output.label_ids

predicted_ids = np.argmax(logits, axis=-1)

print('\n--- Validation Predictions ---')
print('True IDs:', true_ids.tolist())
print('Predicted IDs:', predicted_ids.tolist())

from sklearn.metrics import (
    accuracy_score,
    classification_report,
)


label_names = training_dataset['validation'].features['label'].names
label_ids = list(range(len(label_names)))

print('\n--- Validation Evaluation ---')

print(
    'Accuracy:',
    accuracy_score(true_ids, predicted_ids),
)

print(
    classification_report(
        true_ids,
        predicted_ids,
        labels=label_ids,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
)
from sklearn.metrics import accuracy_score, f1_score


accuracy = accuracy_score(true_ids, predicted_ids)

macro_f1 = f1_score(
    true_ids,
    predicted_ids,
    labels=label_ids,
    average='macro',
    zero_division=0,
)

print('\n--- Evaluation Summary ---')
print(f'Validation accuracy: {accuracy:.4f}')
print(f'Validation macro F1: {macro_f1:.4f}')
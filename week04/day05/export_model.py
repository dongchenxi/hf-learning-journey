from pathlib import Path

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / 'week04'
    / 'day02'
    / 'outputs'
    / 'checkpoint-24'
)

MODEL_DIR = (
    PROJECT_ROOT
    / 'week04'
    / 'day05'
    / 'model'
)

TOKENIZER_CHECKPOINT = 'distilbert/distilbert-base-uncased'


if not CHECKPOINT_PATH.is_dir():
    raise FileNotFoundError(
        f'Checkpoint not found: {CHECKPOINT_PATH}'
    )

model = AutoModelForSequenceClassification.from_pretrained(
    str(CHECKPOINT_PATH)
)

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_CHECKPOINT
)

model.save_pretrained(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))

print('Model saved to:', MODEL_DIR)
print('Label mapping:', model.config.id2label)

reloaded_model = (
    AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR)
    )
)

reloaded_tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR)
)

print('Reloaded model:', type(reloaded_model).__name__)
print('Reloaded tokenizer:', type(reloaded_tokenizer).__name__)
print('Number of labels:', reloaded_model.config.num_labels)
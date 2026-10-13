import argparse
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_DIR = Path(__file__).resolve().parent / 'model'


def predict(text):
    if not text.strip():
        raise ValueError('Input text must not be empty.')

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR)
    )

    inputs = tokenizer(
        text,
        truncation=True,
        max_length=128,
        return_tensors='pt',
    )

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_id = logits.argmax(dim=-1).item()
    predicted_label = model.config.id2label[predicted_id]

    return predicted_label


def main():
    parser = argparse.ArgumentParser(
        description='Classify a customer support message.'
    )

    parser.add_argument(
        'text',
        help='Customer support message to classify.',
    )

    args = parser.parse_args()

    predicted_label = predict(args.text)

    print('Text:', args.text)
    print('Predicted label:', predicted_label)


if __name__ == '__main__':
    main()
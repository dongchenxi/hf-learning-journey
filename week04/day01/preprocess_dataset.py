from collections import Counter
from pathlib import Path
from datasets import load_dataset,DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding



PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / 'week03' / 'day05' / 'data' / 'customer_support.csv'

VALID_LABELS = {
    'shipping',
    'refund',
    'payment',
    'product_issue',
}

dataset = load_dataset(
    'csv',
    data_files=str(DATA_PATH),
    split='train',
)

def has_valid_example(example):
    text = example['text']
    label = example['label']

    return (
        text is not None
        and text.strip() != ''
        and label in VALID_LABELS
    )

def remove_duplicate_texts(dataset):
    seen_texts = set()
    keep_indices = []

    for index, text in enumerate(dataset['text']):
        if text not in seen_texts:
            seen_texts.add(text)
            keep_indices.append(index)

    return dataset.select(keep_indices)

# 1. 清洗无效样本
clean_dataset = dataset.filter(has_valid_example)

# print('Before cleaning:', dataset.num_rows)
# print('After cleaning:', clean_dataset.num_rows)
# print('Clean label counts:', Counter(clean_dataset['label']))

# 2. 检查重复数量
duplicate_count = (
    clean_dataset.num_rows
    - len(set(clean_dataset['text']))
)

# print('Duplicate text count:', duplicate_count)

# 3. 去重
deduplicated_dataset = remove_duplicate_texts(clean_dataset)
#
# print('After deduplication:', deduplicated_dataset.num_rows)
# print('Final label counts:', Counter(deduplicated_dataset['label']))

#ClassLabel 编码
encoded_dataset = deduplicated_dataset.class_encode_column('label')

first_split = encoded_dataset.train_test_split(
    test_size=0.4,
    seed=42,
    stratify_by_column='label',#每个 split 尽量维持各类别(数字)比例
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

# print(dataset_dict)
TOKENIZER_NAME = 'distilbert/distilbert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME
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

training_dataset = tokenized_dataset.remove_columns([
    'text',
    'token_type_ids',
])

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
)
examples = [
    training_dataset['train'][index]
    for index in range(4)
]
batch = data_collator(examples)
print('Batch keys:', batch.keys())
for key, value in batch.items():
    print(key, value.shape)
# print('\nTokenized Dataset:')
# print(tokenized_dataset)
# print('\n--- Split details ---')
#
# for split_name,split_dataset in tokenized_dataset.items():#依次取出每个 split 的“名称 + 对应 Dataset”
#     first_example=split_dataset[0]
#     print(f'\n[{split_name}]')
#     print('Column names:', split_dataset.column_names)
#     print('Features:', split_dataset.features)
#     print('First example:', first_example)
#     token_lengths = [
#         len(input_ids)
#         for input_ids in split_dataset['input_ids']
#     ]
#
#     print(
#         'Token length range:',
#         f'{min(token_lengths)} - {max(token_lengths)}',
#     )
#     print('Decoded text:')
#     print(
#         tokenizer.decode(
#             first_example['input_ids'],
#             skip_special_tokens=True,
#         )
#     )
#
#

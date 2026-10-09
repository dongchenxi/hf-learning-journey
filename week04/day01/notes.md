# Week 4 · Day 1

## Chapter 3：数据预处理与 Train / Validation / Test

## 1. 完整数据流程

```text
Raw CSV
  ↓
load_dataset()
  ↓
Inspect
  ↓
Clean
  ↓
Deduplicate
  ↓
ClassLabel 编码
  ↓
train / validation / test
  ↓
Tokenizer
  ↓
input_ids + attention_mask
  ↓
DataCollatorWithPadding
  ↓
Training-ready Dataset
```

---

## 2. 数据检查

加载数据：

```python
dataset = load_dataset(
    'csv',
    data_files=str(DATA_PATH),
    split='train',
)
```

需要检查：

```python
print(dataset.column_names)
print(dataset.features)
print(dataset.num_rows)
print(dataset[:5])
```

重点确认：

```text
text 列存在
label 列存在
text 不是空值
label 属于预期类别
是否存在重复文本
类别数量是否合理
```

---

## 3. 数据清洗

有效样本需要满足：

```text
text 存在
text 去除首尾空格后不为空
label 存在
label 属于合法类别
```

代码：

```python
VALID_LABELS = {
    'shipping',
    'refund',
    'payment',
    'product_issue',
}


def has_valid_example(example):
    text = example['text']
    label = example['label']

    return (
        text is not None
        and text.strip() != ''
        and label in VALID_LABELS
    )


clean_dataset = dataset.filter(has_valid_example)
```

清洗的核心目的：

```text
删除空文本
删除缺失 label 的样本
删除非法类别
```

---

## 4. 去重

重复文本可能导致数据泄漏：

```text
同一文本 → train
同一文本 → test
```

去重代码：

```python
def remove_duplicate_texts(dataset):
    seen_texts = set()
    keep_indices = []

    for index, text in enumerate(dataset['text']):
        if text not in seen_texts:
            seen_texts.add(text)
            keep_indices.append(index)

    return dataset.select(keep_indices)
```

原则：

```text
同一条文本只能出现在一个样本中。
```

---

## 5. Train / Validation / Test 的职责

### Train

用于：

```text
训练模型
计算 loss
更新模型参数
```

### Validation

用于：

```text
训练过程中的评估
比较模型
选择超参数
选择 checkpoint
```

### Test

用于：

```text
训练和模型选择完成后的最终评估
```

核心区别：

```text
train       → 模型学习
validation  → 开发阶段做选择
test        → 最终正式验收
```

不能根据 test 结果反复修改模型，否则会产生间接数据泄漏。

---

## 6. ClassLabel 编码

原始 label：

```text
'payment'
'product_issue'
'refund'
'shipping'
```

编码：

```python
encoded_dataset = deduplicated_dataset.class_encode_column(
    'label'
)
```

编码后可能是：

```text
0 → payment
1 → product_issue
2 → refund
3 → shipping
```

类别名称保存在：

```python
encoded_dataset.features['label'].names
```

模型训练通常需要整数形式的 label。

---

## 7. 分层切分

当前数据量较小，因此使用：

```text
train       60%
validation  20%
test        20%
```

第一次切分：

```python
first_split = encoded_dataset.train_test_split(
    test_size=0.4,
    seed=42,
    stratify_by_column='label',
)
```

得到：

```text
train       60%
temporary   40%
```

第二次切分：

```python
validation_test_split = first_split['test'].train_test_split(
    test_size=0.5,
    seed=42,
    stratify_by_column='label',
)
```

重新组成：

```python
from datasets import DatasetDict


dataset_dict = DatasetDict({
    'train': first_split['train'],
    'validation': validation_test_split['train'],
    'test': validation_test_split['test'],
})
```

`seed=42` 用于保证切分结果可复现。

---

## 8. 检查类别分布

```python
def get_label_counts(split):
    label_feature = split.features['label']

    readable_labels = [
        label_feature.int2str(label_id)
        for label_id in split['label']
    ]

    return Counter(readable_labels)


for split_name, split in dataset_dict.items():
    print(
        split_name,
        split.num_rows,
        get_label_counts(split),
    )
```

当前结果：

```text
train       47
validation  16
test        16
```

总数：

```text
47 + 16 + 16 = 79
```

不同 split 之间类别尽量保持接近即可，不需要为了练习强行复制样本。

---

## 9. 检查数据泄漏

```python
train_texts = set(dataset_dict['train']['text'])
validation_texts = set(dataset_dict['validation']['text'])
test_texts = set(dataset_dict['test']['text'])

print(len(train_texts & validation_texts))
print(len(train_texts & test_texts))
print(len(validation_texts & test_texts))
```

三个结果都应为：

```text
0
```

表示不同 split 之间没有重复文本。

---

## 10. Tokenization

创建 Tokenizer：

```python
tokenizer = AutoTokenizer.from_pretrained(
    'distilbert-base-uncased'
)
```

定义批处理函数：

```python
def tokenize_function(batch):
    return tokenizer(
        batch['text'],
        truncation=True,
        max_length=128,
    )
```

应用到三个 split：

```python
tokenized_dataset = dataset_dict.map(
    tokenize_function,
    batched=True,
)
```

Tokenization 后新增：

```text
input_ids
attention_mask
token_type_ids
```

不同文本的 token 长度可以不同。

---

## 11. Padding 与 Truncation

### truncation

```python
truncation=True
```

作用：

```text
超过 max_length 时截断输入
```

### padding

今天没有在 Tokenizer 中直接写：

```python
padding=True
```

因为训练时使用动态 Padding。

---

## 12. 移除训练不需要的字段

对于 DistilBERT：

```python
training_dataset = tokenized_dataset.remove_columns([
    'text',
    'token_type_ids',
])
```

保留：

```text
input_ids
attention_mask
label
```

`text` 是原始文本，模型训练时不直接使用。

---

## 13. 动态 Padding

创建 Data Collator：

```python
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
)
```

组成一个 batch：

```python
examples = [
    training_dataset['train'][index]
    for index in range(4)
]

batch = data_collator(examples)
```

最终 batch：

```python
{
    'input_ids': ...,
    'attention_mask': ...,
    'labels': ...,
}
```

当前实验结果：

```text
input_ids      torch.Size([4, 10])
attention_mask torch.Size([4, 10])
labels         torch.Size([4])
```

含义：

```text
4  → batch size
10 → 当前 batch 中最长序列的长度
```

Padding 的规则：

```text
input_ids 中的 0
→ padding token

attention_mask 中的 0
→ 忽略 padding

attention_mask 中的 1
→ 真实 token
```

---

## 14. Dataset Tokenization 与 Tensor 的区别

Dataset Tokenization 阶段：

```python
input_ids
attention_mask
```

通常仍然保存为列表。

组成 batch 时：

```python
DataCollatorWithPadding
```

才完成：

```text
动态 Padding
Tensor 化
label 重命名为 labels
```

因此：

```python
return_tensors='pt'
```

不需要在 Dataset Tokenization 阶段立即使用。

---

## 15. 最终验收检查

```python
expected_columns = {
    'input_ids',
    'attention_mask',
    'label',
}

for split_name, split in training_dataset.items():
    assert set(split.column_names) == expected_columns
    assert split.num_rows > 0

assert batch['input_ids'].shape[0] == 4
assert batch['attention_mask'].shape == batch['input_ids'].shape
assert batch['labels'].shape[0] == 4

print('Final preprocessing checks passed.')
```

---

## 16. 今天最核心的 10 句话

1. 数据处理应遵循 `Load → Inspect → Clean → Split → Tokenize`。
2. `train` 用于更新模型参数。
3. `validation` 用于开发阶段的评估和模型选择。
4. `test` 用于最终评估，不能反复用于调参。
5. 清洗时需要同时检查文本和 label。
6. 重复文本可能造成数据泄漏。
7. `stratify_by_column='label'` 用于保持类别分布。
8. `Dataset.map(batched=True)` 用于批量 Tokenization。
9. `DataCollatorWithPadding` 在组成 batch 时执行动态 Padding。
10. `input_ids`、`attention_mask`、`labels` 是模型训练 batch 的核心字段。

---

## 17. Git Commit

```bash
git status
git diff -- week04/day01
git add week04/day01
git commit -m "study: complete week 4 day 1 data preprocessing"
```

提交前确认没有：

```text
API token
.env
模型权重
缓存文件
临时输出
```
# Week 3 Day 5

## Complete Dataset Pipeline

Raw CSV / JSON
↓
load_dataset()
↓
Dataset
↓
Inspect
↓
Clean
↓
Split
↓
DatasetDict
├── train
├── validation
└── test
↓
map(tokenize_function, batched=True)
↓
Tokenizer
↓
input_ids + attention_mask
↓
Tokenized DatasetDict


## Train / Validation / Test

Train
→ 用于训练模型、更新模型参数

Validation
→ 用于训练过程中评估模型、比较和选择模型/超参数

Test
→ 用于训练和选择完成后的最终评估


## Dataset Card

Dataset Card
=
Dataset 的说明文档，用于描述 Dataset 的用途、
字段、类别、数据划分和限制等信息。


## Dataset Repo

优先发布可复用的原始/清洗 Dataset。

Tokenized Dataset 中的 input_ids 等字段依赖特定 Tokenizer。


## Important

Text Encoding
≠
Label Encoding

Tokenizer:
Text → input_ids

Label Encoding:
Human Label → Class ID
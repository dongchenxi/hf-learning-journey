## Core Memory

- Batch = 一次处理多条数据，提高 GPU 计算效率。
- Padding = 用 `[PAD]` 补齐句子长度，使 Batch 中所有序列长度一致。
- Truncation = 将超过模型最大长度的 Token 截断。
- Attention Mask = 告诉模型哪些 Token 是真实输入，哪些 Token 是 Padding。
- `attention_mask = 1` 表示真实 Token。
- `attention_mask = 0` 表示 Padding Token。
- `return_tensors="pt"` 用于将 Tokenizer 输出转换为 PyTorch Tensor。
- Transformer 的输入通常具有形状：

```
(batch_size, sequence_length)
```

[ ] 理解 Batch
一次性输入多个文本（Text）给模型，提高 GPU 的计算效率（并行计算）。

[ ] 理解 Padding
在较短的文本后添加 [PAD]，使 Batch 中所有序列长度一致。

[ ] 理解 Truncation
当输入文本超过模型支持的最大 Token 数（max_length）时，自动截断超出的 Token。

[ ] 理解 Attention Mask
用于区分哪些是真实 Token，哪些是 Padding Token，Padding Token 不参与注意力计算。

[ ] 知道为什么 Batch 能提高 GPU 利用率
GPU 可以一次并行计算多个样本，提高计算效率。

[ ] 理解为什么 Tensor 要求 Batch 中序列长度一致
Tensor 是规则矩阵，Batch 中所有序列长度必须一致，才能组成 Tensor 并进行计算。

[ ] 理解 padding=True、padding="max_length" 的区别
padding=True：自动补齐到当前 Batch 中最长序列的长度。
padding="max_length"：补齐到指定的 max_length 长度。

[ ] 理解 truncation=True 与 max_length 的关系
max_length 决定模型允许的最大输入长度。
truncation=True 表示输入超过 max_length 时自动截断。

[ ] 理解 attention_mask 中 1 和 0 的含义
1：真实 Token。
0：Padding Token。

[ ] 能解释为什么 Padding 后必须配合 Attention Mask
防止模型将 Padding Token 参与注意力计算，影响模型计算结果。

[ ] 理解 return_tensors="pt" 的作用
将 Tokenizer 的输出（如 input_ids、attention_mask）转换为 PyTorch Tensor，作为模型输入。

[ ] 能解释 torch.Size([batch_size, sequence_length]) 的含义
batch_size：Batch 中包含的文本数量。
sequence_length：Padding（和必要时 Truncation）后，每个序列的统一长度。
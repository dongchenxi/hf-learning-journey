# Week02 Day03：Tokenizer

## Tokenizer

Tokenizer 的作用是将文本转换为模型可以处理的数据。

完整流程：

Text
↓
Tokenization（切分）
↓
Add Special Tokens（添加特殊 Token）
↓
Convert Token → Token ID
↓
Generate input_ids
↓
Convert to Tensor（可选）

---

## input_ids

input_ids 是 Token 对应的整数编号。
Token ID 是 Token 在词表中的唯一整数编号。

模型真正接收的是 Tensor 形式的 input_ids，而不是原始文本。

---

## Special Tokens

Special Tokens 是具有特殊含义的 Token。

常见的包括：

- `[CLS]`：分类任务的起始 Token
- `[SEP]`：句子分隔 Token
- `[PAD]`：补齐长度 Token
- `[MASK]`：掩码 Token（BERT 预训练）
- `[UNK]`：未知 Token

---

## Decode

decode() 用于将 Token ID 转换回文本。

设置：

```python
skip_special_tokens=True
```

可以去掉 `[CLS]`、`[SEP]` 等特殊 Token。

---

## 今日总结

Tokenizer 不只是把文本切成 Token，它还负责：

- 文本切分（Tokenization）
- 添加 Special Tokens
- Token 转换为 Token ID
- 生成 input_ids
- （可选）转换为 Tensor
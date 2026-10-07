## Core Memory

- Batch = 一次处理多条数据，提高 GPU 计算效率。
- Padding = 用 `[PAD]` 补齐句子长度，使 Batch 中所有序列长度一致，补到当前 batch 最长序列。
- Truncation = 将超过模型最大长度的 Token 截断。
- Attention Mask = 告诉模型哪些 Token 是真实输入，哪些 Token 是 Padding。
- `attention_mask = 1` 表示真实 Token。
- `attention_mask = 0` 表示 Padding Token。
- `return_tensors="pt"` 用于将 Tokenizer 输出转换为 PyTorch Tensor。
- Transformer 的输入通常具有形状：
input_ids
[B, L]

attention_mask
[B, L]

Sequence Classification logits
[B, C]

1. **Batch 和 Batch size 分别是什么？为什么深度学习通常使用 Batch？**
batch是一次性处理多个文本
batch size是一个batch中包含的文本条数
2. **为什么一个 Batch 中的序列通常需要相同长度？**
多条数据组成规则的tensor作为模型的输入
3. **Padding 是什么？为什么需要 Padding？`[PAD]` 是什么？**
Padding补齐
保证每条batch文本的长度是一致的
[PAD]是补齐的特殊token
4. **`padding=True` 做什么？它和 `padding="max_length"` 有什么区别？**
padding=True是补齐到batch中最长文本的长度
padding="max_length补齐到指定文本长度
5. **为什么有时候多个文本组成 Batch，即使没有写 `padding=True` 也不会报错？**
tokenizer的时候，多个文本的长度恰好是一致的
6. **Truncation 是什么？为什么需要 Truncation？`truncation=True` 做什么？**
截长
输入序列的长度超过模型的最长token输入，需要截断
tokenizer的时候，输入序列的长度超过模型的最长token输入，进行截断
7. **`max_length` 限制的是字符数、单词数，还是 tokenizer 最终构造的输入序列长度？**
tokenizer 最终构造的输入序列长度
8. **Padding 和 Truncation 有什么区别？**
补齐
截断
9. **`attention_mask` 是什么？其中的 `1` 和 `0` 分别表示什么？既然 `input_ids` 中已经有 `[PAD]`，为什么还需要 `attention_mask`？为什么它通常和 `input_ids` shape 相同？**
attention_mask告诉模型哪些是真实的token，哪些是Padding的token
1真实的token
0Padding的token
需要告诉模型哪些是真实的token，哪些是padding的token，所以还需要attention_mask
input_ids和attention_mask是一一对应的
10. **如何理解常见 Tensor Shape？**
   ```text
   input_ids.shape = [4, 10]（b=4,l=10）
   attention_mask.shape = [4, 10]（b=4,l=10）
   logits.shape = [4, 2]（b=4,c=10）
   ```
   对 Sequence Classification 来说，每个维度分别代表什么？


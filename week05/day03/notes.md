# Week 5 Day 3 — Chat Template

## 1. Chat Template

Chat Template 是将结构化对话消息转换为符合特定模型要求的文本格式的模板。

不同模型可能使用不同的 Chat Template。

## 2. apply_chat_template()

```python
formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

核心参数：

- `tokenize=False`：返回格式化后的文本字符串。
- `tokenize=True`：应用 Chat Template，并将结果转换为 Token IDs。
- `add_generation_prompt=True`：按照模型的 Chat Template，在末尾添加开始生成 Assistant 回复所需的提示格式。

## 3. 不同模型的聊天格式

### Qwen2.5-Instruct

```text
<|im_start|>system
You are a Python teacher.<|im_end|>
<|im_start|>user
What is Python?<|im_end|>
<|im_start|>assistant
```

### Zephyr

```text
<|system|>
You are a Python teacher.</s>
<|user|>
What is Python?</s>
<|assistant|>
```

核心区别：

- Qwen 使用 `<|im_start|>` 和 `<|im_end|>`。
- Zephyr 使用角色标记和 `</s>`。

## 4. Token IDs

使用 `tokenize=True` 可以获得 Token IDs。

本次实验实际返回：

```python
{
    "input_ids": [...],
    "attention_mask": [...]
}
```

Qwen2.5 的部分特殊 Token：

- `151644`：`<|im_start|>`
- `151645`：`<|im_end|>`

实验结果：

```text
Same text: True
```

说明本次实验中，Token IDs 解码后的文本与格式化文本一致。

## 5. 实验结论

1. Chat Template 将结构化消息转换为模型要求的聊天格式。
2. 不同模型可能使用不同的角色标记和消息边界。
3. `tokenize=False` 用于观察格式化文本。
4. `tokenize=True` 用于生成 Token IDs。
5. 应使用与目标模型匹配的 Chat Template。

本日实验只使用 Tokenizer，没有加载模型权重或执行模型推理。
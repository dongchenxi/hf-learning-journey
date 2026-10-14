# Week 6 Day 3: 创建并检查 LoRA Adapter

## 实际实验配置

- 基础模型：`Qwen/Qwen2.5-0.5B-Instruct`
- `r=8`
- `lora_alpha=16`，缩放系数 `alpha / r = 2`
- `target_modules=["q_proj", "v_proj"]`
- `lora_dropout=0.1`
- `bias="none"`
- `task_type="CAUSAL_LM"`

使用 `get_peft_model(model, lora_config)` 后，PEFT 会在目标线性层加入 LoRA Adapter，并返回 PEFT 模型。

## 参数统计

实验结果：

```text
trainable params: 540,672
all params:       494,573,440
trainable%:       0.1093
```

原始模型参数被冻结；新增的 LoRA 参数 `lora_A` 和 `lora_B` 可训练。因此，只训练约 `0.11%` 的参数即可为任务做适配。

## 第 0 个 Transformer Block 的形状与计算

| 模块 | LoRA A 形状 | LoRA B 形状 | 新增参数 |
| --- | --- | --- | ---: |
| `q_proj` | `(8, 896)` | `(896, 8)` | `8 x 896 + 896 x 8 = 14,336` |
| `v_proj` | `(8, 896)` | `(128, 8)` | `8 x 896 + 128 x 8 = 8,192` |

两者的秩 `r` 都是 8，但 `v_proj` 的输出维度为 128、`q_proj` 的输出维度为 896，所以 `lora_B` 形状不同，新增参数数量也不同。

模型有 24 个 Transformer Block；每个 Block 对 `q_proj` 与 `v_proj` 各加入一个 Adapter，共 48 个目标模块。单层新增参数为 `14,336 + 8,192 = 22,528`，总计为 `22,528 x 24 = 540,672`。

## Adapter 与训练的区别

创建 Adapter 不等于完成微调：`get_peft_model()` 只负责按 LoRA 配置添加可训练的低秩矩阵 A、B，并冻结原始模型参数。完成训练仍需执行前向计算、损失计算、反向传播与优化器更新。

一句话概括：**LoRA Adapter 是加在预训练模型目标模块上的可训练低秩矩阵 A、B，让模型在冻结原始参数时学习任务适配。**

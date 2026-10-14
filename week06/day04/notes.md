# Week 6 Day 4：训练并保存 LoRA Adapter

## 1. 学习目标与实际配置

本日完成一个最小 LoRA 训练闭环：准备聊天格式训练数据、执行前向与反向传播、更新 LoRA 参数，并保存 Adapter 与实验日志。

| 项目 | 实际值 |
| --- | --- |
| 基础模型 | `Qwen/Qwen2.5-0.5B-Instruct` |
| LoRA 秩 `r` | `8` |
| `lora_alpha` | `16`（缩放系数 `16 / 8 = 2`） |
| 目标模块 | `q_proj`、`v_proj` |
| `lora_dropout` | `0.1` |
| Batch size | `1` |
| 学习率 | `1e-4` |
| 优化器 | `AdamW` |
| 随机种子 | `42` |

LoRA 只训练新增的低秩矩阵，基础模型参数保持冻结：

```text
trainable params: 540,672
all params:       494,573,440
trainable%:       0.1093%
```

## 2. 训练数据准备

训练集包含 3 条中文电商问答样本：查询订单、申请退款和修改收货地址。每条样本依次经过：

```text
messages → apply_chat_template → tokenizer → input_ids / attention_mask / labels
```

本练习中 `labels = input_ids.clone()`，因此损失覆盖整段聊天文本。这适合演示 Causal LM 的基本训练流程；更接近指令微调实践时，通常会将用户提示和模板标记位置设为 `-100`，只对助手回答计算损失。

## 3. 训练流程

每个 epoch 按样本逐条训练，核心顺序为：

```python
outputs = peft_model(..., labels=labels)  # Forward，并得到 loss
loss.backward()                           # Backward，计算梯度
optimizer.step()                          # 更新可训练 LoRA 参数
optimizer.zero_grad()                     # 清空已累积的梯度
```

训练运行 3 个 epoch；每个 epoch 有 3 个 batch，因此总步数为 `3 × 3 = 9`。训练损失从 `4.3579` 降至 `3.2662`：

```text
4.3579 → 4.2982 → 3.8478 → 3.8227 → 3.7948
       → 3.4765 → 3.4862 → 3.4914 → 3.2662
```

为确认优化器确实更新了 LoRA 参数，脚本在训练前使用 `detach().clone()` 保存一个 `lora_B` 参数快照。训练后，跟踪参数
`base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight`
的最大绝对变化为 `0.0009044055`，所以 `lora_parameter_changed=True`。

## 4. 保存结果与版本控制

训练脚本调用：

```python
peft_model.save_pretrained(ADAPTER_DIR)
```

在 `outputs/lora_adapter/` 本地生成 `adapter_model.safetensors`、`adapter_config.json` 和 README；其中 `adapter_model.safetensors` 保存训练后的 LoRA 权重。该目录属于可重新生成的模型产物，已加入 `.gitignore`，不纳入本次 Git 提交。

`outputs/training_log.json` 则纳入版本控制，记录模型与 LoRA 配置、训练设置、9 步完整 loss、以及参数变化检查。二者不能互相替代：

- Adapter 是训练成果，用于复用已学到的 LoRA 增量；
- JSON 日志是实验记录，用于追溯配置与分析结果。

下次推理必须加载 **基础模型 + LoRA Adapter**；Adapter 本身不包含完整基础模型权重。

## 5. 局限与下一步验证方向

这只是 3 个样本、9 步的教学实验。训练 loss 下降说明模型对这批训练序列的目标有所拟合，但不足以证明泛化能力或回答质量。这里尚未进行独立验证集评估、生成质量对比、prompt-only loss masking，或保存后重新加载 Adapter 的推理检查；这些应在后续实验中单独验证。

## 6. 本日结论

1. LoRA 训练的完整闭环是 Forward → Loss → Backward → Optimizer step → Zero grad。
2. `model.train()` 切换训练行为，`requires_grad` 决定参数是否可求梯度；二者不能互相替代。
3. 用 `detach().clone()` 保存训练前快照，才能可靠验证参数是否变化。
4. Adapter 保存可复用的权重，训练日志保存实验可追溯性；重新推理需要 Base Model + Adapter。

# Week 4 · Day 2 核心笔记

## 1. 从数据到训练

```text
Tokenized Dataset
→ DataCollatorWithPadding
→ Tensor Batch
→ 分类模型
→ loss / logits
→ Trainer 执行训练
→ validation 评估
→ 选择最佳 checkpoint
```

## 2. 分类模型与类别映射

```python
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=4,
    id2label=id2label,
    label2id=label2id,
)
```

- `num_labels`：类别数量。
- `id2label`：整数 ID 转类别名称。
- `label2id`：类别名称转整数 ID。
- 模型与 Tokenizer 应使用匹配的 checkpoint。

本次映射：

```text
0 → payment
1 → product_issue
2 → refund
3 → shipping
```

基础 checkpoint 的分类头可能需要新初始化，所以微调前的预测尚未具备客服分类能力。

## 3. Forward Pass、Logits 与 Loss

```python
model.eval()

with torch.no_grad():
    outputs = model(**batch)
```

`model(**batch)` 将 batch 字典展开为模型输入参数。

```text
logits shape = [batch_size, num_labels]
```

本次：

```text
[4, 4] → 四条文本，每条四个类别分数
```

logits 是原始分数，可以为负数。

```python
predicted_ids = outputs.logits.argmax(dim=-1)
```

对每条文本选择分数最高的类别。

```python
probabilities = torch.softmax(outputs.logits, dim=-1)
```

将分数转换为每条文本合计约为 1 的类别概率。

传入 `labels` 时，模型可以计算分类 loss；本次不传 `labels` 时，`loss` 为 `None`，仍有 logits。

`model.eval()` 切换评估模式；`torch.no_grad()` 关闭梯度记录。这次观察输出没有更新参数。

## 4. TrainingArguments 与 Trainer

```text
TrainingArguments → 定义训练设置
Trainer           → 执行训练、评估和保存
trainer.train()   → 开始训练
```

本次主要设置：

```text
learning_rate                  2e-5
per_device_train_batch_size     4
num_train_epochs               3
eval_strategy                  epoch
save_strategy                  epoch
metric_for_best_model          accuracy
load_best_model_at_end         True
```

Trainer 在训练过程中组织 batch、执行前向计算、反向传播和参数更新，并按配置评估与保存。

```python
train_dataset=training_dataset['train']
eval_dataset=training_dataset['validation']
```

validation 用于选择 checkpoint，test 保留用于最终评估。

## 5. Epoch、Batch 与 Step

- Batch：一次送入模型的一组样本。
- Epoch：完整遍历训练集一次。
- Optimizer step：一次模型参数更新。

本次单设备、梯度累积为 1：

```text
47 条训练样本，batch size 为 4
→ 每个 epoch 12 个 batch
→ 3 个 epoch 共 36 次更新
```

最后一个 batch 可以不足四条。

## 6. 如何看训练日志

- `loss`：最近日志区间内的平均训练 loss。
- `train_loss`：整个训练过程的汇总训练 loss。
- `eval_loss`：评估数据上的 loss。
- `eval_accuracy`：评估样本中预测正确的比例。
- `learning_rate`：当前更新使用的学习率，会随调度器变化。
- `grad_norm`：梯度范数，用于观察梯度规模。

训练 loss 局部波动正常，需要结合整体趋势与 validation 判断。

loss 下降时，accuracy 可能不变：真实类别的概率提高了，但最高分类别没有改变。

## 7. 本次实验结果

| 阶段 | Validation loss | Accuracy |
|---|---:|---:|
| 训练前 | 1.3898 | 25% |
| Epoch 1 | 约 1.363 | 43.75% |
| Epoch 2 | 约 1.334 | 62.5% |
| Epoch 3 | 约 1.316 | 62.5% |

本次按 accuracy 选择模型，Epoch 2 和 Epoch 3 并列，最终恢复了：

```text
checkpoint-24
```

同一个训练 batch：

```text
正确数：1/4 → 3/4
loss：1.3885 → 1.3099
```

训练 batch 的改善说明模型适应了这些样本；validation 用于观察泛化表现。validation 只有 16 条，每条影响 6.25 个百分点，结果仍容易波动。

## 8. 复现与设备

```python
set_seed(42)
# 然后创建模型
```

随机种子需要在分类头初始化之前设置。相同 seed 有助于复现，但不同设备或软件环境仍可能存在差异。

本次 MPS 的 pin-memory 警告没有阻止训练，可以设置：

```python
dataloader_pin_memory=False
```

## 今天最核心的五句话

1. 分类模型输出 logits，提供真实标签后还可以计算 loss。
2. Forward pass 本身不等于参数更新，训练还需要反向传播和优化器。
3. TrainingArguments 配置训练，Trainer 执行训练。
4. 用 validation 指标选择 checkpoint，最后恢复的模型不一定来自最后一个 epoch。
5. 训练样本上的改善与泛化表现需要分别观察。
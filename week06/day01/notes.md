# Week 6 Day 1：Full Fine-tuning 与 PEFT

## 1. 学习目标

理解 Full Fine-tuning 与 Parameter-Efficient Fine-tuning（PEFT）的区别，
并通过一个简单的 PyTorch 模型观察参数冻结、反向传播和优化器更新的行为。

本日只建立 PEFT 的基础概念；LoRA 的具体结构和超参数留到后续学习。

## 2. Full Fine-tuning 与 PEFT

### Full Fine-tuning

Full Fine-tuning 会让预训练模型的全部参数参与训练：

- 所有参数通常都设置为 `requires_grad=True`；
- 反向传播会为全部可训练参数计算梯度；
- 优化器会更新全部参数。

它的表达能力强，但对大模型来说会消耗更多显存、训练时间和存储空间。

### PEFT

PEFT 的核心是在保留大部分预训练参数不变的前提下，只训练少量参数，
或训练额外添加的小型参数模块。

常见思路是：

- 冻结大部分原始参数；
- 只让少量参数参与梯度计算和优化；
- 用更少的训练资源适配新任务。

## 3. `requires_grad` 与参数冻结

`requires_grad` 决定 PyTorch 是否需要为一个参数计算梯度：

```python
for param in model.layer1.parameters():
    param.requires_grad = False
```

设置为 `False` 后，该参数被冻结：

- `loss.backward()` 后，其 `param.grad` 为 `None`；
- 即使该参数出现在模型中，优化器也不应更新它；
- 冻结不等于从前向计算中移除，它仍可参与前向计算。

`requires_grad=True` 表示允许计算梯度；参数还必须参与当前计算图，才会得到实际梯度。

## 4. 实验 1：冻结参数与统计

文件：`01_freeze_parameters.py`

模型结构：

```text
Linear(10, 20) → Linear(20, 5)
```

冻结 `layer1` 后的参数统计：

| 项目 | 数量 |
| --- | ---: |
| Total Parameters | 325 |
| Trainable Parameters（`layer2`） | 105 |
| Frozen Parameters（`layer1`） | 220 |
| Trainable Ratio | 32.31% |

参数数量的计算：

- `layer1.weight`: `20 × 10 = 200`，`layer1.bias`: `20`，共 `220`；
- `layer2.weight`: `5 × 20 = 100`，`layer2.bias`: `5`，共 `105`。

这个小实验模拟了 PEFT 的关键思想：只有未冻结的参数是可训练参数。

## 5. Loss、反向传播与优化器

第二个实验文件为 `02_backward_optimizer.py`。

训练的基本顺序是：

```python
output = model(x)
loss = loss_fn(output, target)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

| 步骤 | 作用 |
| --- | --- |
| Forward | 使用输入数据得到模型预测 `output`。 |
| Loss | 衡量预测值与真实目标 `target` 的差异。 |
| `optimizer.zero_grad()` | 清除上一次保留的梯度。 |
| `loss.backward()` | 通过反向传播计算可训练参数的梯度。 |
| `optimizer.step()` | 根据梯度更新优化器管理的参数。 |

本实验使用 `nn.MSELoss()`。例如预测为 `[2, 4]`、目标为 `[3, 6]` 时：

```text
MSE = ((2 - 3)^2 + (4 - 6)^2) / 2 = 2.5
```

由于 `layer1` 已冻结，反向传播后 `layer1` 的梯度为 `None`；`layer2` 允许求导且参与了计算图，因此会获得梯度。

优化器通过以下方式只接收可训练参数：

```python
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.01,
)
```

## 6. 引用与独立副本

下面两种写法看起来相似，但含义不同：

```python
before[name] = param
```

这会保存对原参数对象的引用。执行 `optimizer.step()` 更新参数后，
`before[name]` 会看到同一个对象的更新值；因此再与当前参数比较时，
可能错误地得到 `changed=False`。

```python
before[name] = param.detach().clone()
```

这会保存一个独立的训练前快照：

- `detach()`：使快照脱离当前计算图；
- `clone()`：复制底层数据，避免与原参数共享同一份数据。

因此，正确检查更新应使用：

```python
before = {
    name: param.detach().clone()
    for name, param in model.named_parameters()
}
```

本日运行的 `02_backward_optimizer.py` 使用了引用写法，其 `changed=False` 输出展示的正是这个陷阱；它不表示 `layer2` 没有被优化器更新。

最小例子：

```python
a = torch.tensor([10.0])
b = a
c = a.clone()
a[0] = 20.0
```

结果是 `b[0] == 20.0`，因为 `b` 与 `a` 引用同一张量；
`c[0] == 10.0`，因为 `c` 是修改前创建的独立副本。

## 7. 本日结论

1. Full Fine-tuning 更新模型全部参数；PEFT 只训练少量参数或附加模块。
2. `requires_grad=False` 冻结参数，使其不计算梯度也不被优化器更新。
3. Loss 衡量预测与目标的差异；`backward()` 计算梯度；`step()` 更新参数。
4. 比较更新前后的参数时，必须用 `detach().clone()` 保存独立快照，不能只保存引用。
5. 后续教学模型统一使用 `Qwen/Qwen2.5-0.5B`，设备为 M5 32GB。

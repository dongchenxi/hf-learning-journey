# Week 6 Day 2：LoRA 核心超参数

## 1. 学习目标

理解 LoRA（Low-Rank Adaptation）的低秩更新形式，以及四个常用配置项：
`r`、`lora_alpha`、`target_modules` 和 `lora_dropout`。

本日实验使用 `Qwen/Qwen2.5-0.5B-Instruct` 检查模块名称；不进行训练，也不进入 Day 3。

## 2. LoRA 的基本思路

对于原始线性层权重 `W`，LoRA 冻结预训练权重，并学习一个低秩增量：

```text
W' = W + ΔW
ΔW = B A
```

其中，若原线性层输入维度为 `d_in`、输出维度为 `d_out`，则低秩矩阵可写为：

```text
A: r × d_in
B: d_out × r
```

原始线性层权重有 `d_in × d_out` 个参数；LoRA 分支有：

```text
r × (d_in + d_out)
```

个可训练参数。因此当 `r` 远小于层的维度时，LoRA 能显著减少训练参数量。

## 3. `r`：低秩维度

`r` 是 LoRA 的秩（rank），决定低秩矩阵的中间维度，直接影响可训练参数数量和分支的表达能力。

实验文件：`01_lora_rank.py`

对一个 `512 → 512` 线性层，原始权重参数量为 `512 × 512 = 262,144`。实验结果：

| `r` | LoRA 参数量 | 占原始权重比例 |
| ---: | ---: | ---: |
| 4 | 4,096 | 1.56% |
| 8 | 8,192 | 3.12% |
| 16 | 16,384 | 6.25% |
| 32 | 32,768 | 12.50% |

在层维度不变时，`r` 翻倍，LoRA 参数量也翻倍。

## 4. `lora_alpha`：更新缩放

标准 LoRA 会对分支输出使用缩放系数：

```text
scaling = lora_alpha / r
output = base_output + scaling × lora_output
```

`lora_alpha` 控制 LoRA 更新注入原模型输出时的相对幅度。固定 `r` 时，增大 `lora_alpha` 会增大缩放系数；固定 `lora_alpha` 时，增大 `r` 会减小缩放系数。

实验文件：`02_lora_alpha.py`。在 `base_output=10`、`lora_output=2` 时：

| `r` | `alpha` | 缩放系数 | 最终输出 |
| ---: | ---: | ---: | ---: |
| 8 | 8 | 1.00 | 12.00 |
| 8 | 16 | 2.00 | 14.00 |
| 8 | 32 | 4.00 | 18.00 |
| 16 | 16 | 1.00 | 12.00 |
| 16 | 32 | 2.00 | 14.00 |

例如只把 `r` 从 8 改成 16、保持 `lora_alpha=16`，缩放系数会从 `16 / 8 = 2` 变为 `16 / 16 = 1`，即减半；同时每个已应用 LoRA 的线性层的 LoRA 参数量翻倍。

## 5. `target_modules`：应用 LoRA 的位置

`target_modules` 指定模型中哪些模块接收 LoRA 分支。它决定的是被改造的层集合，而不是单个层的秩或缩放。

实验文件：`03_inspect_target_modules.py`。对 `Qwen/Qwen2.5-0.5B-Instruct` 的线性层名称进行检查后：

| 配置 | `target_modules` | 匹配模块数 |
| --- | --- | ---: |
| A | `q_proj`, `v_proj` | 48 |
| B | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 96 |

该模型有 24 个 Transformer 层，因此配置 A 中每层匹配 `q_proj` 与 `v_proj` 两个模块，共 `24 × 2 = 48` 个。仅改变 `r` 不会改变 `target_modules` 的匹配数量。

## 6. `lora_dropout`：LoRA 分支的训练期随机失活

`lora_dropout` 仅作用于 LoRA 分支的输入（或分支计算），不改变原始冻结权重分支。

在训练模式下，`lora_dropout=0.1` 表示每个输入元素独立地以 10% 的概率被置为 0；并不表示每一次前向传播都恰好置零 10% 的元素。Dropout 会对保留下来的元素按实现规则缩放，以保持期望值。

实验文件：`04_lora_dropout.py`，其 `p=0.5` 演示了：

- `train()` 模式下：连续两次前向传播采用不同随机掩码，输出可能不同；
- `eval()` 模式下：Dropout 不再随机置零，同一输入的两次输出一致，且与输入相同。

## 7. 配置综合理解

```python
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)
```

- `r=8`：低秩维度为 8，控制每个 LoRA 分支的参数量与容量；
- `lora_alpha=16`：标准缩放为 `16 / 8 = 2`；
- `target_modules`：在每层的查询投影和价值投影上应用 LoRA；
- `lora_dropout=0.1`：训练时 LoRA 分支输入元素以 10% 概率置零。

## 8. 本日结论

1. LoRA 用低秩矩阵学习增量，冻结原始权重，只训练少量附加参数。
2. `r` 越大，LoRA 参数量与表达能力通常越大；在其他设置不变时，参数量随 `r` 线性增加。
3. 标准缩放为 `lora_alpha / r`，因此不能只孤立地理解 `r` 或 `lora_alpha`。
4. `target_modules` 决定 LoRA 放在哪些模块上；改变 `r` 不会改变其匹配模块数。
5. `lora_dropout` 仅在训练模式下给 LoRA 分支引入随机失活，评估模式下不生效。

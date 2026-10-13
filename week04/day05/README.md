# Customer Support Text Classification

使用 DistilBERT 对英文客服问题进行四分类，包含数据预处理、微调、评估、错误分析和命令行推理。

## 分类类别

| 类别 | 含义 |
|---|---|
| payment | 支付失败、付款方式、交易或扣款问题 |
| product_issue | 商品损坏、缺件、功能或质量问题 |
| refund | 退款申请、退货后的退款、退款进度或金额问题 |
| shipping | 配送、物流追踪、收货地址问题 |

## 数据与训练

数据来自本地客服练习数据集：

```text
week03/day05/data/customer_support.csv
```

清洗并去重后共有 79 条样本，采用分层切分：

| Split | 样本数 |
|---|---:|
| train | 47 |
| validation | 16 |
| test | 16 |

基础模型为 `distilbert/distilbert-base-uncased`。

训练配置：

```text
Learning rate: 2e-5
Batch size: 4
Epochs: 3
Seed: 42
```

每个 epoch 在 validation 上评估一次，按照 validation accuracy 选择 checkpoint。本次选择了 `checkpoint-24`。

## 安装依赖

以下命令从学习仓库根目录运行：

```bash
python -m pip install -r week04/day05/requirements.txt
```

实验环境依赖版本：

```text
torch==2.14.1
transformers==5.19.0
datasets==5.1.0
accelerate==1.15.0
scikit-learn==1.9.1
```

依赖版本来自完成实验时的虚拟环境，尚未在全新环境中验证。

## 导出模型

已有 Day 2 的 `checkpoint-24` 后运行：

```bash
python week04/day05/export_model.py
```

模型和 Tokenizer 保存到：

```text
week04/day05/model/
```

此步骤依赖本地训练 checkpoint。模型尚未发布到 Hugging Face，单独克隆代码仓库不会获得模型权重。

## 推理

```bash
python week04/day05/inference.py "Where is my package?"
```

本次输出：

```text
Text: Where is my package?
Predicted label: shipping
```

推理流程：

```text
Text → Tokenizer → Model → Logits → Predicted label
```

## Validation 评估结果

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| payment | 0.0000 | 0.0000 | 0.0000 | 4 |
| product_issue | 0.8000 | 1.0000 | 0.8889 | 4 |
| refund | 0.3750 | 0.7500 | 0.5000 | 4 |
| shipping | 1.0000 | 0.7500 | 0.8571 | 4 |

整体指标：

```text
Validation Accuracy: 0.6250
Validation Macro F1: 0.5615
```

以上是 validation 指标。Test 尚未进行最终评估。

## 错误分析

已分析 validation 中全部 6 条错误：

- 4 条 payment 被预测为 refund。
- 1 条 shipping 被预测为 refund。
- 1 条 refund 被预测为 product_issue。

例如：

```text
Text: Can I use another payment method?
True label: payment
Predicted label: refund
```

该问题询问付款方式，属于 payment。模型将其误判为 refund，显示出两类之间的区分不足。

## 当前局限与后续工作

- 数据集仅有 79 条样本，validation 每类只有 4 条，指标容易波动。
- 模型未正确识别 validation 中的 payment 样本。
- 已完成 6 个错误案例分析，尚未达到学习路线要求的至少 20 个。
- 后续补充训练样本时，保留现有 validation，避免改变比较基准。
- 后续完成模型发布和独立的 test 最终评估。

本项目用于展示文本分类的完整学习流程，当前效果尚不足以支持实际客服自动分流。

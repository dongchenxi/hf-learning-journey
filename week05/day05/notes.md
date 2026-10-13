# Week 5 Day 5：Base Model Baseline

## 1. 学习目标

建立一个可复现、可验证、可用于后续模型比较的
Base Model Baseline。

## 2. 核心概念

### 2.1 Model Baseline

Baseline 是在固定实验条件下记录的模型表现，
用于后续比较。

建立 Baseline 需要：

- 固定测试集
- 明确模型及其版本
- 固定生成参数
- 保存模型回答和实验信息
- 验证数据完整性与对齐关系

### 2.2 固定评测集

固定评测集是预先确定、在后续评估中保持不变的一组
测试样本，用于比较不同模型或方法的表现。

本次构建了 40 条电商客服 Prompt，覆盖 8 个类别。

每条样本包含：

- id：唯一标识测试样本
- category：业务类别
- prompt：测试问题

### 2.3 Model Revision

Model Revision 是模型仓库的版本标识，
用于确定实验使用的具体模型版本。

只记录模型名称可能导致后续实验使用不同版本，
影响结果的可复现性。

### 2.4 数据对齐验证

数据对齐验证是指检查不同数据集之间的关联字段
是否一致，确保每条结果对应正确的输入样本。

不仅需要验证 ID 一致，还需要验证 Prompt 一致。

## 3. 核心推理流程

Prompt
→ Tokenizer
→ input_ids
→ Base Model
→ generate()
→ generated_ids
→ decode()
→ response

对于当前 Decoder-only Causal LM：

input_length = inputs["input_ids"].shape[1]

generated_ids = outputs[0][input_length:]

第一行获取输入 Prompt 的 Token 数量。

第二行去掉输入 Prompt，只保留新生成的 Token。

## 4. 推理模式

model.eval()：

将模型切换到评估模式。

torch.inference_mode()：

在模型推理时关闭自动梯度计算，
并减少额外的计算开销。

两者作用不同，不能相互替代。

## 5. 五个实验之间的关系

实验 1：验证单条推理流程。

实验 2：构建 40 条固定测试 Prompt。

实验 3：复用单条推理流程，批量生成并保存结果。

实验 4：验证结果数量、字段、ID 和 Prompt 对齐。

实验 5：固定模型 Revision，重新生成并验证结果。

完整流程：

固定测试集
→ 固定模型及生成条件
→ 批量推理
→ 保存结果和实验元数据
→ 验证数据完整性与对齐关系
→ 建立 Baseline

## 6. 实验配置

Model：

Qwen/Qwen2.5-0.5B

Revision：

060db6499f32faf8b98477b0a26969ef7d8b9987

Generation Config：

- max_new_tokens = 64
- do_sample = False

## 7. 实验结果

测试集：

data/test_prompts.jsonl

Baseline：

results/base_baseline.jsonl

实际结果：

- 测试样本：40 条
- Baseline 结果：40 条
- ID 唯一性：PASS
- 必要字段：PASS
- ID 对齐：PASS
- Prompt 对齐：PASS
- 模型、Revision 和生成参数：PASS
- Response 类型：PASS

最终验证：

Baseline validation passed!

## 8. 后续模型比较

比较 Base Model 与 Instruct Model 时：

保持一致：

- 测试问题
- 生成参数（尽可能保持一致）
- 数据验证规则
- 评估标准

需要改变：

- 模型及其对应 Revision
- 模型要求的输入格式（例如 Chat Template）

公平比较的核心原则：

只改变需要研究的实验变量，
其他条件尽可能保持一致。

## 9. 注意事项

数据完整性验证不等于回答质量评估。

当前 Baseline 已保存模型回答，
但尚未建立自动化的回答质量评分标准。
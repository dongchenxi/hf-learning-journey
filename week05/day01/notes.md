| 模型类型 | 核心定义 |
|---|---|
| **Base Model** | 主要通过 Next Token Prediction 进行预训练，学习语言规律和知识。 |
| **Instruct Model** | 在预训练模型基础上，进一步通过指令与回答数据进行训练，使模型更好地遵循用户指令。 |
| **Chat Model** | 面向多轮对话场景，能够利用角色信息和对话历史生成回复；与 Instruct Model 并非完全互斥的类别。 |

# Week 5 Day 1 — Base Model、Instruct Model 与 Chat Model

## 1. 核心概念

### Base Model

主要通过 Next Token Prediction 进行预训练，
学习语言规律和知识。

Base Model 也可能生成正确的问答内容，
但能够回答问题不等于经过 Instruction Fine-tuning。

### Instruct Model

在预训练模型基础上，进一步通过指令与回答数据进行训练，
使模型更好地遵循用户指令。

### Chat Model

面向多轮对话场景，能够利用角色信息和对话历史生成回复。

Chat Model 与 Instruct Model 并非完全互斥的类别。

## 2. 实验

模型：

- Qwen/Qwen2.5-0.5B
- Qwen/Qwen2.5-0.5B-Instruct

相同输入：

What is Python?

生成参数：

- max_new_tokens=20
- do_sample=False

### Base Model 输出

Python is a high-level, interpreted, general-purpose
programming language. It is known for its simplicity,

### Instruct Model 输出

Python is a high-level, interpreted language

## 3. 实验结论

两个模型都能回答问题。

Base Model 能够回答问题，
不代表它经过了 Instruction Fine-tuning。

本次实验只比较了两个模型在同一原始文本 Prompt 下的输出，
没有使用 Chat Template。

因此，不能仅凭本次结果判断两个模型的完整指令遵循能力。
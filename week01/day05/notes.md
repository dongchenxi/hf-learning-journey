Hugging Face 生态图
    hub-AI 平台
        models
        datasets
        spaces
    libraries-开发生态,python库
        datasets
        transformers
        huggingface_hub

推理流程
Raw Text
原始文本
   ↓
Tokenizer
文本 → Token IDs
   ↓
Model
模型推理
   ↓
Raw Output
原始输出
   ↓
Postprocessing
后处理
   ↓
Readable Result
可读结果
# Week 1 Day 5 - Weekly Review

## Core Relationship / 核心关系

- **Hugging Face Hub = 托管模型、数据集、应用的平台**
- **Transformers = 加载和使用 Transformer 模型的 Python 库**
- **Tokenizer = Text → Tokens → Token IDs**
- **Model = 执行神经网络计算**
- **Pipeline = preprocessing + inference + postprocessing**

## Transformer Architectures / Transformer 架构

- **Encoder = Understand = 理解**
- **Decoder = Generate = 生成**
- **Encoder-Decoder = Understand then Generate = 理解后生成**

- **BERT = Encoder-only**
- **GPT = Decoder-only**
- **T5 = Encoder-Decoder**

## Generation Parameters / 生成参数

- **max_new_tokens = 生成多长**
- **temperature = 多随机**
- **top_k = 留多少候选 token**
- **top_p = 留多大的累计概率范围**

## Week 1 Result / 第一周结果

- [ ] 能解释 Hub / Transformers / Model / Tokenizer / Pipeline
- [ ] 能在 3 分钟内解释 Encoder / Decoder / Encoder-Decoder
- [ ] 不看教程完成最小模型推理
## Core Memory / 核心记忆

- **Encoder = Understand = 理解**
- **Decoder = Generate = 生成**
- **Encoder-Decoder = Understand then Generate = 理解后生成**

- **BERT = Encoder-only = 理解类任务**
- **GPT = Decoder-only = 通过不断预测下一个 token 来生成文本**
- **T5 = Encoder-Decoder = 理解输入后生成输出**
## Interview Practice / 双语面试练习

### Q1. What is the difference between BERT and GPT?
### BERT 和 GPT 的主要区别是什么？

**English**

BERT is mainly an encoder-only model and is commonly used for understanding tasks such as classification and NER.

GPT is decoder-only and generates text by repeatedly predicting the next token.

**中文**

BERT 主要是 Encoder-only，适合理解类任务，例如文本分类和 NER。

GPT 是 Decoder-only，主要通过不断预测下一个 token 来生成文本。

---

### Q2. Why is BERT suitable for text classification?
### 为什么 BERT 适合文本分类？

**English**

BERT is good at understanding the context of the input sequence, so it works well for classification tasks.

**中文**

BERT 擅长理解整个输入序列的上下文，所以非常适合文本分类任务。

---

### Q3. Why is GPT suitable for text generation?
### 为什么 GPT 适合文本生成？

**English**

GPT predicts the next token based on previous tokens and repeats this process to generate text.

**中文**

GPT 根据前面的 token 预测下一个 token，并不断重复这个过程生成文本。

---

### Q4. Why is T5 suitable for translation or summarization?
### 为什么 T5 适合翻译或摘要？

**English**

T5 uses an encoder-decoder architecture. The encoder understands the input, and the decoder generates the output.

**中文**

T5 使用 Encoder-Decoder 架构。Encoder 负责理解输入，Decoder 负责生成输出。

---

### Q5. What are the three main Transformer architecture types?
### Transformer 三种主要架构是什么？

**English**

The three main types are encoder-only, decoder-only, and encoder-decoder.

**中文**

三种主要架构是 Encoder-only、Decoder-only 和 Encoder-Decoder。
**能解释 Padding 为什么需要 Attention Mask？**
答案：
Attention Mask 用于标记 Batch 中哪些是真实 Token，哪些是 Padding Token。
其中：
- 1：真实 Token
- 0：Padding Token
模型在计算 Attention 时会忽略 Padding Token，避免 Padding Token 参与计算，从而保证模型计算结果的准确性。
一句话总结：
Padding 用于统一序列长度，Attention Mask 用于让模型忽略 Padding Token，两者通常需要配合使用。

**能解释 AutoModel 与 AutoModelForXXX 的差异**
答案：
AutoModel 是 Hugging Face 提供的自动模型加载器，用于加载 Base Model，不包含任何 Task Head，输出 Hidden States，主要用于特征提取（Feature Extraction）。
AutoModelForXXX 是在 Base Model 的基础上增加对应的 Task Head，用于完成具体任务，例如文本分类、文本生成等。
一句话总结：
AutoModel 输出 Hidden States；AutoModelForXXX 在 Base Model 上增加 Task Head，用于完成具体任务。
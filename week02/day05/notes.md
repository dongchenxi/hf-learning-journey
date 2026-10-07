## Core Memory

- Batch = 一次处理多条数据。
- Batch size = 一个 Batch 中包含的样本数量。

- Padding = 用 `[PAD]` 补齐句子长度，使 Batch 中所有序列长度一致。
- `padding=True` = 补到当前 Batch 最长序列。

- Attention Mask = 告诉模型哪些 Token 是真实输入，哪些 Token 是 Padding。
- `attention_mask = 1` 表示真实 Token。
- `attention_mask = 0` 表示 Padding Token。

- Truncation = 将超过模型最大长度的 Token 截断。

- `return_tensors="pt"` 用于将 Tokenizer 输出转换为 PyTorch Tensor。

- AutoModel 加载 Base Transformer，主要输出基础表示。
- AutoModelForXXX 在 Base Transformer 上增加对应任务的 Head，用于完成具体任务。

- AutoModelForSequenceClassification
  = Base Transformer + Sequence Classification Head。

- Sequence Classification 手动推理流程：

Text
→ Tokenizer
→ input_ids + attention_mask
→ Model
→ Logits
→ Softmax
→ Probabilities
→ Argmax
→ Class ID
→ id2label
→ Prediction

- Tensor Shape：

input_ids
[B, L]

attention_mask
[B, L]

Sequence Classification logits
[B, C]

B = Batch Size
L = Sequence Length
C = Number of Classes

**不使用 pipeline() 时，Sequence Classification 的完整批量推理流程是什么？**
texts
tokenizer->inputid Attention Mask
models
Logits
softmax
Probabilities
Argmax
列表ID
ID2label
Prediction
**为什么 Batch 推理通常需要 Padding？Padding 后为什么还需要 Attention Mask？**
保证batch中的文本长度一致
需要告诉模型哪些是真实的token，哪些是padding的token
**AutoModel 和 AutoModelForXXX 的核心区别是什么？为什么当前任务使用 AutoModelForSequenceClassification？**
AutoModel加载base model，输出last hidden state
AutoModelForXXX 加载base model，并加上对应的任务头，
当前任务是 Sequence Classification，因此使用 AutoModelForSequenceClassification。
**outputs.logits → softmax → argmax → id2label 分别完成什么工作？**
outputs.logits--每个类别的预测分数
softmax--输出预测概率
argmax--找到最大概率的class ID
id2label--ID转换为人类可读的标签
**批量推理中，为什么不能把整个 predicted_class_ids 直接作为一个类别 ID 使用？如何得到每条文本对应的 label 和 score？**
predicted_class_ids
→ 整个 Batch 的 Class IDs

class_id
→ 当前一个样本的 Class ID

probs
→ 当前一个样本所有类别的概率

probs[class_id]
→ 当前样本预测类别对应的概率
**如何把批量预测结果组织成 text / label / score，并分别输出为 JSON 和 CSV？**
results=[]
resulta.append(text / label / score)
import JSON/csv,将对应的结果写入到对应的格式
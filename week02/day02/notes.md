**AutoModel**
from transformers import AutoModel

model = AutoModel.from_pretrained(
    model_name
)
作用是加载基础的transformer模型
# print(outputs.last_hidden_state.shape)
# torch.Size([2, 7, 768]) 2条text，分别切割成7个token,Token 在最后一层得到的向量表示

**AutoModelForSequenceClassification**
from transformers import (
    AutoModelForSequenceClassification
)

作用是加载SequenceClassification模型
详细见day01的笔记


**AutoModelForCausalLM**
from transformers import AutoModelForCausalLM

作用是加载CausalLM模型
核心任务是：
previous tokens
      ↓
predict next token

logits shape：
[
 batch_size,
 sequence_length,
 vocabulary_size（vocabulary_size = 模型可以选择的 token 候选数量）
]
预测下一个token
next_token_logits = outputs.logits[:, -1, :]
next_token_id = torch.argmax(
    next_token_logits,
    dim=-1,
)

next_token = tokenizer.decode(
    next_token_id
)

**AutoModel 是什么？**  
加载基础 Transformer 模型，主要输出每个 Token 的向量表示。

**AutoModelForSequenceClassification 是什么？**  
用于文本分类的模型 = 基础模型 + 分类头。

**AutoModelForCausalLM 是什么？**  
用于预测下一个 Token、生成文本的模型 = 基础模型 + LM Head。

**Auto 是什么意思？**  
根据模型配置，自动选择正确的模型类型。

**AutoModel 和 AutoModelForXXX 的核心区别是什么？**  
`AutoModel`：基础模型，没有具体任务头。  
`AutoModelForXXX`：基础模型 + 特定任务 Head。

**什么是 task head？**  
接在基础模型后面，负责完成具体任务的部分。

**Classification Head 是干什么的？**  
把模型得到的向量转换成各类别的预测分数。

**LM Head 是干什么的？**  
把模型得到的向量转换成词表中各 Token 的预测分数。

**last_hidden_state 是什么？**  
每个 Token 经过 Transformer 最后一层后得到的向量。

**last_hidden_state 的三个维度是什么？**

```text
[一次几条文本, Token 数量, 每个 Token 的向量长度]
[batch_size, sequence_length, hidden_size]
```

**Sequence Classification 的 logits 表示什么？**  
一条文本属于每个类别的原始预测分数。

**Causal LM 的 logits 表示什么？**  
每个位置对“下一个 Token 是谁”的原始预测分数。

**为什么两种 logits 的 shape 不一样？**  
因为分类是在**整句话中选类别**，Causal LM 是在**每个位置预测 Token**。

```text
分类：[batch_size, 类别数量]

Causal LM：[batch_size, Token数量, 词表大小]
```

**Causal LM 为什么可以用来生成文本？**  
不断预测下一个 Token：

```text
我喜欢
→ 吃
→ 苹果
→ 。
```

一个一个预测下去，就生成了文本。

**class_id 和 token_id 有什么区别？**  
`class_id`：类别的编号，例如 `0 = NEGATIVE`。  
`token_id`：Token 的编号，例如 `"hello" → 7592`。
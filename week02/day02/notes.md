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
不改了，就记这个版本

文本
 ↓
Tokenizer
 ↓
input_ids / attention_mask
 ↓
PyTorch Tensor
 ↓
Transformer Model
 ↓
logits
 ↓
softmax
 ↓
probabilities
 ↓
Argmax
   ↓
Class ID
   ↓
id2label
   ↓
Prediction
对 NLP sequence classification(文本分类任务) 来说，首先由 tokenizer 对输入文本进行 tokenization，并转换成 input_ids、attention_mask 等模型输入 Tensor。
随后这些 Tensor 被传入 Transformer sequence-classification model 做 forward pass，得到每个类别对应的 logits。
对于单标签分类，可以对 logits 应用 softmax 得到类别概率，再通过 argmax 得到预测类别 ID，最后利用模型 config 中的 id2label 映射成人类可读的 label。

Tokenizer是分词器，作用是将text转换成input_ids / attention_mask
Model神经网络计算，将Tensor做前向传播，得到每个类别的logits
Forward Pass，向前计算，outputs = model(**inputs)，输入数据经过模型向前计算，得到输出的过程
logits模型对每个类别产生的原始分数。
Softmax类别概率，logits → class probabilities
Argmax找到概率最大的类别
id2label ID → Label


**AutoTokenizer 是干什么的？**
加载分词器，将文本->token->token id
**input_ids 是什么？**
模型输入的Tensor ID
**attention_mask 是什么？**
告诉模型哪些是真实的token，哪些是padding的token ID
**return_tensors="pt" 是什么？**
让Tokenizer分词器返回pytorh Tensor
_model(**inputs) 为什么有 **？_
将字典inputs拆关键字入参给模型
**forward pass 是什么？**
根据模型输入，向前计算得到模型输出
**outputs.logits 是什么？**
每个类别的原始预测分数
****logits 为什么不是 probability？**
logits是每个类别的原始预测分数，没有经过归一化计算得到probability
**softmax 做什么？**
logits经过函数softmax，计算得到probability
**dim=-1 是什么意思？**
最后一个维度的计算
**argmax 做什么？**
最大值
**model.config.id2label 做什么？**
token ID-人类可读的label
**为什么 inference 使用 torch.no_grad()？**
关闭梯度计算，减少内存，提高计算效率
**pipeline 到底替我们封装了哪些步骤？**

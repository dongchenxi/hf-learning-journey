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


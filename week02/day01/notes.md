# Week02 Day01：Behind the Pipeline

### Tokenizer过程及对应的名词解释
文本（Text）
↓（Tokenization，切分）
Token
↓（Token ID）
转变为Tensor

## input_ids（Tokenizer 的最终输出）
Token对应的整数编号。

## Tensor
模型的输入，PyTorch 中最核心的数据结构。

## Model
根据输入计算输出。

## Logits
模型输出的原始分数（raw scores）。

## Softmax
Softmax 把 Logits 转换成概率。
### 1. `pipeline()` 为什么叫高级 API？它封装了哪些步骤？

因为 `pipeline()` 对底层推理流程进行了封装，用户只需输入文本即可完成推理。

封装的主要步骤：
```text
加载 Tokenizer
↓
文本 Tokenization
↓
转换为 Tensor
↓
加载 Model
↓
模型前向推理（Forward Pass）
↓
得到 Logits
↓
Softmax 转换为概率
↓
输出 Prediction（Label）
```

---

### 2. Tokenizer 的职责是什么？为什么模型不能直接处理字符串？

**Tokenizer 的职责：** 将文本转换为模型可以处理的数字表示。

模型不能直接处理字符串，因为神经网络只能计算数字（Tensor），不能计算文本。

---

### 3. 什么是 Token？它为什么不等于“一个单词”？

**Token** 是模型处理文本的最小单位。

它不一定等于一个单词，一个单词可能会被拆成多个 Token，一个 Token 也可能是一个汉字、一个词或一个词的一部分。

---

### 4. 什么是 `input_ids`？模型真正接收到的是什么？

**input_ids** 是 Token 对应的整数编号，是 Tokenizer 的最终输出。

模型真正接收到的是 **Tensor 形式的 `input_ids`**，而不是原始文本。

---

### 5. 为什么要把 `input_ids` 转成 Tensor？

因为 PyTorch 模型只能处理 **Tensor**。

Tensor 是 PyTorch 中最核心的数据结构，也是模型的输入格式。

---

### 6. 什么是 Forward Pass（前向推理）？

Forward Pass（前向推理）是指**将输入送入模型，计算得到输出**的过程。

---

### 7. 什么是 Logits？它为什么不是概率？

**Logits** 是模型最后一层神经网络计算出来的一组原始分数（Raw Scores）。

它不是概率，因为各个值没有经过归一化，总和不一定等于 1。

---

### 8. 为什么还需要 Softmax？`argmax` 又在做什么？

**Softmax** 用于将 Logits 转换成概率分布。

**argmax** 用于找到概率最大的类别索引，并作为最终预测结果。
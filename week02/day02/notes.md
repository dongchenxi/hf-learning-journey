| 类 | 输出 | 使用场景 |
|----|------|----------|
| AutoModel | Hidden States | 特征提取 |
| AutoModelForSequenceClassification | Logits | 分类 |
| AutoModelForCausalLM | Generated Text | 文本生成 |


### 1. 什么是 `AutoModel`？它为什么叫“Auto”？

`AutoModel` 是 Hugging Face 提供的自动模型加载器（Automatic Model Loader）。

它会根据模型的配置文件（`config.json`）自动识别模型类型，并加载对应的模型类，因此叫 **"Auto"**。

---

### 2. `AutoModel` 和具体模型（如 `BertModel`）是什么关系？

`AutoModel` 是统一的入口，`BertModel`、`RobertaModel` 等是具体的模型实现。

例如：

```python
AutoModel.from_pretrained("bert-base-uncased")
```

内部实际上会自动加载：

```python
BertModel.from_pretrained("bert-base-uncased")
```

---

### 3. 什么是 **Base Model（基础模型）**？

Base Model 是没有添加任何任务头（Task Head）的基础模型。

它只负责提取文本特征，不直接完成分类、生成等具体任务。

---

### 4. 什么是 **Hidden States**？`AutoModel` 为什么输出它？

**Hidden States** 是模型对输入文本学习得到的特征表示（Feature Representation）。

`AutoModel` 的作用是提取特征，因此输出 Hidden States，而不是分类结果或生成文本。

---

### 5. 什么是 **Classification Head**？它解决了什么问题？

**Classification Head** 是添加在 Base Model 后面的分类层（通常是 Linear Layer）。

它负责将 Hidden States 转换为分类任务所需的 Logits，从而完成文本分类。

---

### 6. `AutoModelForSequenceClassification` 与 `AutoModel` 的区别是什么？

- **AutoModel**：输出 Hidden States，用于特征提取。
- **AutoModelForSequenceClassification**：在 Base Model 上增加了 Classification Head，输出 Logits，用于文本分类。

---

### 7. 什么是 **Causal Language Model（因果语言模型）**？为什么叫“Causal”？

Causal Language Model 是一种用于文本生成的语言模型。

之所以叫 **Causal**，是因为模型在生成当前 Token 时，只能利用前面的 Token，不能看到后面的 Token。

---

### 8. 什么情况下应该使用 `AutoModel`、`AutoModelForSequenceClassification`、`AutoModelForCausalLM`？

- **AutoModel**：用于特征提取、Embedding、自定义模型开发。
- **AutoModelForSequenceClassification**：用于文本分类、情感分析等分类任务。
- **AutoModelForCausalLM**：用于文本生成、对话、代码生成等自回归生成任务。
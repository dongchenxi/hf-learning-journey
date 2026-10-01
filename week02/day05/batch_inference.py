from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

#batch
texts = [
    "I love Hugging Face.",
    "This movie is terrible.",
    "The weather is wonderful today.",
    "I am disappointed with the service.",
    "Transformers are amazing."
]
#tokenizer分词器
#from_pretrained一键加载预训练模型的权重和配置
tokenizer=AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
#加载模型
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
#使用分析器处理文本
#return_tensors=用于将 Tokenizer 输出转换为 PyTorch Tensor
#padding=True batch补齐
#truncation自动截断
encoding=tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
# outputs = model(
#     input_ids=encoding["input_ids"],
#     attention_mask=encoding["attention_mask"]
# )
outputs = model(**encoding)
#从 outputs 中提取 logits
# print(outputs.logits)
#使用 Softmax 计算归一化概率
probabilities = torch.softmax(
    outputs.logits,
    dim=-1
)
# print(probabilities)
#获取预测类别 ID（正类别 负类别）
predictions = torch.argmax(probabilities, dim=-1)
# print(predictions)
labels = [
    #model.config.id2label[...]：
    #id2label 是预训练模型配置文件中存储的数字ID到文本标签映射字典。
    model.config.id2label[idx.item()]
    #遍历 0 1
    for idx in predictions
]
results = []
for text, label in zip(
    texts,
    labels
):
    results.append(
        {
            "text": text,
            "label": label
        }
    )

print(results)
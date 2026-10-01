from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
model_name="distilbert-base-uncased-finetuned-sst-2-english"
#from_pretrained() 用于把预训练权重加载进模型。
tokenizer = AutoTokenizer.from_pretrained(model_name)
text= "I love Hugging Face!"
inputs = tokenizer(
    text,
    return_tensors="pt"
)
# print(inputs)#tokenizer之后得到input_ids和attention_mask，函数PyTorch Tensor，作为model输入
model=AutoModelForSequenceClassification.from_pretrained(model_name)
with torch.no_grad():
    outputs = model(**inputs)
# print(outputs)#模型输出logits，模型对每个类别产生的原始分数
# Tensor shape
probabilities = torch.softmax(
    outputs.logits,
    dim=-1
)
print(probabilities)#函数softmax将logits变为概率
predicted_class_id = probabilities.argmax(dim=-1)
print(predicted_class_id.item())#找到概率最大的类别 类别ID
label=model.config.id2label[predicted_class_id.item()]
print(label)

from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
import json
import csv
model_name = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)#将原始文本作为模型的输入，可以将token ID解码回可读性文本的文本处理组件
model = AutoModelForSequenceClassification.from_pretrained(model_name)
texts = [
    "I love this movie!",
    "This movie is terrible.",
    "The movie is okay.",
    "I hate this product.",
]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
# print(inputs.keys())
# input_ids
# → 每个位置是什么 Token
#
# attention_mask
# → 哪些位置是真实输入，哪些位置是 Padding
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits
probabilities=torch.softmax(logits, dim=-1)
predicted_class_ids = torch.argmax(
    probabilities,
    dim=-1,
)
# =====================================
results = []

for text, probs, class_id in zip(
    texts,
    probabilities,
    predicted_class_ids,
):
    class_id = class_id.item()
    label = model.config.id2label[class_id]
    score = probs[class_id].item()
    results.append(
        {
            "text": text,
            "label": label,
            "score": score,
        }
    )

print(results)

with open(
    "predictions.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2,
    )

with open(
    "predictions.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "text",
            "label",
            "score",
        ],
    )

    writer.writeheader()
    writer.writerows(results)
from transformers import  pipeline

classifier=pipeline(
    "sentiment-analysis",
"distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

text = "I really enjoy learning Hugging Face."

result = classifier(text)

print(result)
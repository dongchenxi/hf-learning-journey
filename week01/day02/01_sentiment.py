from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

result = classifier("I really enjoy learning Hugging Face.")

# print(result)
print(type(classifier))
print(type(classifier.model))
print(type(classifier.tokenizer))
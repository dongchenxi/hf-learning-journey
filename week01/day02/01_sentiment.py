# from transformers import pipeline
#
# classifier = pipeline(
#     "sentiment-analysis",
#     model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
# )
#
# result = classifier("I really enjoy learning Hugging Face.")
#
# # print(result)
# print(type(classifier))
# print(type(classifier.model))
# print(type(classifier.tokenizer))

from transformers import pipeline

classifier = pipeline(
    task="sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

texts = [
    "I love this product.",
    "This movie is terrible.",
    "The service was okay.",
]

results = classifier(texts)

for text, result in zip(texts, results):
    print("Text:", text)
    print("Prediction:", result)
    print("-" * 50)
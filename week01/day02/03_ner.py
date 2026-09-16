from transformers import pipeline

ner = pipeline(
    task="ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple",
)

text = "Apple hired John Smith to work in Singapore."

result = ner(text)

for entity in result:
    print(entity)
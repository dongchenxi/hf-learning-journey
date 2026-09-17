from transformers import pipeline, GenerationConfig

generator = pipeline(
    "text-generation",
    model="openai-community/gpt2",
)

prompt = "Artificial intelligence will"

generation_config = GenerationConfig(
    max_new_tokens=30,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
)

result = generator(
    prompt,
    generation_config=generation_config,
)

print(result[0]["generated_text"])
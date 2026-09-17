from transformers import pipeline, GenerationConfig

generator = pipeline(
    "text-generation",
    model="openai-community/gpt2",
)

prompt = "Artificial intelligence will"

temperatures = [0.2, 0.7, 1.2]

for temp in temperatures:
    print("=" * 80)
    print("Temperature:", temp)

    config = GenerationConfig(
        max_new_tokens=30,
        do_sample=True,
        temperature=temp,
        top_p=0.9,
    )

    result = generator(
        prompt,
        generation_config=config,
    )

    print(result[0]["generated_text"])
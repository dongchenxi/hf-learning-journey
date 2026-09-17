from transformers import pipeline, GenerationConfig

generator = pipeline(
    "text-generation",
    model="openai-community/gpt2",
)

prompt = "Machine learning is"

top_p_values = [0.5, 0.8, 0.95]

for top_p in top_p_values:
    print("=" * 80)
    print("top_p:", top_p)

    config = GenerationConfig(
        max_new_tokens=30,
        do_sample=True,
        temperature=0.8,
        top_p=top_p,
    )

    result = generator(
        prompt,
        generation_config=config,
    )

    print(result[0]["generated_text"])
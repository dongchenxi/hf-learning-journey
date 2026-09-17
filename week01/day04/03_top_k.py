from transformers import pipeline, GenerationConfig

generator = pipeline(
    "text-generation",
    model="openai-community/gpt2",
)

prompt = "The future of software engineering is"

top_k_values = [5, 20, 50]

for top_k in top_k_values:
    print("=" * 80)
    print("top_k:", top_k)

    config = GenerationConfig(
        max_new_tokens=30,
        do_sample=True,
        temperature=0.8,
        top_k=top_k,
    )

    result = generator(
        prompt,
        generation_config=config,
    )

    print(result[0]["generated_text"])
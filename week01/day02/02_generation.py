# from transformers import pipeline
#
# generator = pipeline(
#     "text-generation",
#     model="openai-community/gpt2",
# )
#
# result = generator(
#     "Artificial intelligence will",
#     max_new_tokens=30,
# )
#
# print(result)

from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="openai-community/gpt2",
)

prompts = [
    "Artificial intelligence will",
    "Machine learning is",
    "The future of software engineering is",
]

for prompt in prompts:
    result = generator(
        prompt,
        max_new_tokens=30,
    )

    print("Prompt:", prompt)
    print("Result:", result[0]["generated_text"])
    print("-" * 80)



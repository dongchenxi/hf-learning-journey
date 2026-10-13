messages = [
    {
        "role": "system",
        "content": "You are a Python teacher."
    },
    {
        "role": "user",
        "content": "What is Python?"
    },
    {
        "role": "assistant",
        "content": "Python is a programming language."
    },
    {
        "role": "user",
        "content": "Is it easy to learn?"
    }
]

for message in messages:
    print("Role:", message["role"])
    print("Content:", message["content"])
    print("---")
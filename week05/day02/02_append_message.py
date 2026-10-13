messages = [
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

# 模拟模型生成的回复
assistant_reply = "Yes, Python is beginner-friendly."

# 将 Assistant 回复添加到对话历史
messages.append({
    "role": "assistant",
    "content": assistant_reply
})

# 查看更新后的对话历史
for message in messages:
    print(f"{message['role']}: {message['content']}")
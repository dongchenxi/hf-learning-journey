import json


# 1. 普通客服 QA 数据
qa_data = {
    "question": "我的订单什么时候发货？",
    "answer": "您的订单通常会在付款后 24 小时内发货。"
}


# 2. 转换成 Instruction 格式
instruction_data = {
    "instruction": "请回答用户的客服问题。",
    "input": qa_data["question"],
    "output": qa_data["answer"]
}


# 3. 转换成 Messages 格式
messages_data = {
    "messages": [
        {
            "role": "system",
            "content": "你是一名专业的电商客服助手。"
        },
        {
            "role": "user",
            "content": qa_data["question"]
        },
        {
            "role": "assistant",
            "content": qa_data["answer"]
        }
    ]
}


# 4. 查看转换结果
print("=== Original QA ===")
print(json.dumps(qa_data, ensure_ascii=False, indent=2))

print("\n=== Instruction Dataset ===")
print(json.dumps(instruction_data, ensure_ascii=False, indent=2))

print("\n=== Messages Dataset ===")
print(json.dumps(messages_data, ensure_ascii=False, indent=2))
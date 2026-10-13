import json


# 1. 原始 QA 数据集
qa_dataset = [
    {
        "question": "我的订单什么时候发货？",
        "answer": "您的订单通常会在付款后 24 小时内发货。"
    },
    {
        "question": "商品支持七天无理由退货吗？",
        "answer": "符合退货条件的商品支持七天无理由退货。"
    },
    {
        "question": "如何查询物流信息？",
        "answer": "您可以在订单详情页面查看物流信息。"
    }
]


# 2. 准备两个结果列表
instruction_dataset = []
messages_dataset = []


# 3. 批量转换
for qa in qa_dataset:

    instruction_data = {
        "instruction": "请回答用户的客服问题。",
        "input": qa["question"],
        "output": qa["answer"]
    }

    messages_data = {
        "messages": [
            {
                "role": "system",
                "content": "你是一名专业的电商客服助手。"
            },
            {
                "role": "user",
                "content": qa["question"]
            },
            {
                "role": "assistant",
                "content": qa["answer"]
            }
        ]
    }

    instruction_dataset.append(instruction_data)
    messages_dataset.append(messages_data)


# 4. 检查转换结果
print("Original QA count:", len(qa_dataset))
print("Instruction count:", len(instruction_dataset))
print("Messages count:", len(messages_dataset))

print("\n=== First Instruction ===")
print(json.dumps(instruction_dataset[0], ensure_ascii=False, indent=2))

print("\n=== First Messages ===")
print(json.dumps(messages_dataset[0], ensure_ascii=False, indent=2))


from pathlib import Path


# 5. 创建输出目录
output_dir = Path(__file__).resolve().parent / "data"
output_dir.mkdir(parents=True, exist_ok=True)


# 6. 保存 Instruction Dataset
instruction_path = output_dir / "instruction.jsonl"

with open(instruction_path, "w", encoding="utf-8") as f:
    for item in instruction_dataset:
        line = json.dumps(item, ensure_ascii=False)
        f.write(line + "\n")


# 7. 保存 Messages Dataset
messages_path = output_dir / "messages.jsonl"

with open(messages_path, "w", encoding="utf-8") as f:
    for item in messages_dataset:
        line = json.dumps(item, ensure_ascii=False)
        f.write(line + "\n")


print("\n=== Saved Files ===")
print(instruction_path)
print(messages_path)

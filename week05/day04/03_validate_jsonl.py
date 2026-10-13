import json
from pathlib import Path


# 1. 动态获取数据目录
data_dir = Path(__file__).resolve().parent / "data"


# 2. 定义 JSONL 读取函数
def load_jsonl(file_path):
    dataset = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)#将每一行的字符串还原为 Python 字典
            dataset.append(item)

    return dataset


# 3. 读取两个数据集
instruction_dataset = load_jsonl(
    data_dir / "instruction.jsonl"
)

messages_dataset = load_jsonl(
    data_dir / "messages.jsonl"
)


# 4. 检查样本数量
print("Instruction count:", len(instruction_dataset))
print("Messages count:", len(messages_dataset))


# 5. 验证 Instruction 数据结构
for item in instruction_dataset:
    assert set(item.keys()) == {
        "instruction",
        "input",
        "output"
    }

print("Instruction structure: PASS")


# 6. 验证 Messages 数据结构
for item in messages_dataset:
    assert "messages" in item

    messages = item["messages"]

    assert isinstance(messages, list)

    roles = [message["role"] for message in messages]

    assert roles == [
        "system",
        "user",
        "assistant"
    ]

    for message in messages:
        assert isinstance(message["content"], str)

print("Messages structure: PASS")


# 7. 验证原始问题与答案是否一致
for instruction_item, messages_item in zip(
    instruction_dataset,
    messages_dataset
):
    messages = messages_item["messages"]

    assert instruction_item["input"] == messages[1]["content"]
    assert instruction_item["output"] == messages[2]["content"]

print("QA content consistency: PASS")
import json
from pathlib import Path


# 1. 固定实验配置
EXPECTED_MODEL = "Qwen/Qwen2.5-0.5B"

EXPECTED_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"

EXPECTED_CONFIG = {
    "max_new_tokens": 64,
    "do_sample": False,
}


# 2. 文件路径
BASE_DIR = Path(__file__).resolve().parent

TEST_PATH = BASE_DIR / "data" / "test_prompts.jsonl"
RESULT_PATH = BASE_DIR / "results" / "base_baseline.jsonl"


# 3. 读取 JSONL 文件
def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


test_samples = load_jsonl(TEST_PATH)
results = load_jsonl(RESULT_PATH)


# 4. 验证样本数量
assert len(test_samples) == 40
assert len(results) == 40

print("Sample count: PASS")


# 5. 验证 ID 唯一性
test_ids = [item["id"] for item in test_samples]
result_ids = [item["id"] for item in results]

assert len(test_ids) == len(set(test_ids))
assert len(result_ids) == len(set(result_ids))

print("Unique IDs: PASS")


# 6. 验证必要字段
required_fields = {
    "id",
    "category",
    "prompt",
    "model",
    "revision",
    "generation_config",
    "response",
}

for result in results:
    assert required_fields <= result.keys()

print("Required fields: PASS")


# 7. 根据 ID 建立测试集索引
test_by_id = {
    item["id"]: item
    for item in test_samples
}


# 8. 验证测试集与结果集的 ID 一致
assert set(test_ids) == set(result_ids)

print("ID alignment: PASS")


# 9. 验证每条 Baseline 结果
for result in results:

    # 根据 ID 查找对应的测试样本
    sample = test_by_id[result["id"]]

    # 验证输入 Prompt 和类别一致
    assert result["prompt"] == sample["prompt"]
    assert result["category"] == sample["category"]

    # 验证模型名称
    assert result["model"] == EXPECTED_MODEL

    # 验证模型版本
    assert result["revision"] == EXPECTED_REVISION

    # 验证生成参数
    assert result["generation_config"] == EXPECTED_CONFIG

    # 验证模型回答的数据类型
    assert isinstance(result["response"], str)


print("Prompt alignment: PASS")
print("Model, revision and config: PASS")
print("Response type: PASS")


# 10. 验证完成
print("\nBaseline validation passed!")
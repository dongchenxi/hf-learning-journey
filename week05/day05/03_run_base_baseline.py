import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# 1. 固定实验配置
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
GENERATION_CONFIG = {
    "max_new_tokens": 64,
    "do_sample": False,
}


# 2. 文件路径
BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "data" / "test_prompts.jsonl"
OUTPUT_PATH = BASE_DIR / "results" / "base_baseline.jsonl"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# 3. 读取固定测试集
with INPUT_PATH.open("r", encoding="utf-8") as f:
    test_samples = [json.loads(line) for line in f if line.strip()]

assert len(test_samples) == 40


# 4. 加载模型
print("Loading model:", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
)

model.eval()


# 5. 逐条生成并保存
with OUTPUT_PATH.open("w", encoding="utf-8") as f:

    for index, sample in enumerate(test_samples, start=1):

        prompt = sample["prompt"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        )

        with torch.inference_mode():

            outputs = model.generate(
                **inputs,
                **GENERATION_CONFIG,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 去掉输入 Prompt，只保留新生成的 Token
        input_length = inputs["input_ids"].shape[1]

        generated_ids = outputs[0][input_length:]

        response = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        # 保存测试样本、模型信息和生成结果
        result = {
            "id": sample["id"],
            "category": sample["category"],
            "prompt": prompt,
            "model": MODEL_NAME,
            "revision": MODEL_REVISION,
            "generation_config": GENERATION_CONFIG.copy(),
            "response": response,
        }

        f.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )

        # 每条结果及时写入磁盘
        f.flush()

        print(
            f"[{index}/{len(test_samples)}] "
            f"{sample['id']} completed"
        )


print("\nBaseline generation completed!")
print("Saved to:", OUTPUT_PATH)
import torch

from collections import Counter
from transformers import AutoModelForCausalLM


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
)

# 2. 找出模型中的线性层
linear_modules = [
    name
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Linear)
]

# 3. 统计线性层名称
module_counts = Counter(
    name.split(".")[-1]
    for name in linear_modules
)

print("\n=== Linear Module Names ===")

for name, count in sorted(module_counts.items()):
    print(f"{name:<20} count={count}")


# 4. 比较两组 target_modules
configs = {
    "A": ["q_proj", "v_proj"],
    "B": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
}

for config_name, targets in configs.items():

    matched = [
        name
        for name in linear_modules
        if name.split(".")[-1] in targets
    ]

    print(f"\n=== Config {config_name} ===")
    print("target_modules:", targets)
    print("matched modules:", len(matched))

    print("examples:")

    for name in matched[:5]:
        print(" ", name)
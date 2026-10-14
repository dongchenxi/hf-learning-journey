import json
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


# ============================================================
# 1. 基础配置
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

LEARNING_RATE = 1e-4
NUM_EPOCHS = 3

# 使用脚本所在目录构建输出路径，避免受运行位置影响
DAY_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = DAY_DIR / "outputs"
ADAPTER_DIR = OUTPUT_DIR / "lora_adapter"
LOG_PATH = OUTPUT_DIR / "training_log.json"

torch.manual_seed(42)


# ============================================================
# 2. 准备训练数据
# ============================================================

examples = [
    {
        "question": "如何查询订单？",
        "answer": "您可以在个人中心的订单页面查询订单状态。",
    },
    {
        "question": "如何申请退款？",
        "answer": "您可以进入订单详情页面，点击申请退款。",
    },
    {
        "question": "如何修改收货地址？",
        "answer": "您可以在订单发货前进入订单详情页面修改收货地址。",
    },
]


# ============================================================
# 3. 加载 Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Tokenizer loaded.")


# ============================================================
# 4. 加载 Base Model
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
)

print("Base Model loaded.")


# ============================================================
# 5. 创建 LoRA Adapter
# ============================================================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(
    model,
    lora_config,
)

peft_model.print_trainable_parameters()


# ============================================================
# 6. 准备训练数据：Chat Template + Tokenization
# ============================================================

train_dataset = []

for example in examples:

    messages = [
        {
            "role": "user",
            "content": example["question"],
        },
        {
            "role": "assistant",
            "content": example["answer"],
        },
    ]

    # 6.1 应用 Chat Template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # 6.2 Tokenization
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # 6.3 创建 Labels
    labels = input_ids.clone()

    # 6.4 保存训练样本
    train_dataset.append(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    )

print(f"Training examples: {len(train_dataset)}")


# ============================================================
# 7. 创建 Optimizer
# ============================================================

optimizer = AdamW(
    filter(
        lambda p: p.requires_grad,
        peft_model.parameters(),
    ),
    lr=LEARNING_RATE,
)


# ============================================================
# 8. 保存训练前的 LoRA 参数
# ============================================================

lora_param_name = None
lora_param_before = None

for name, param in peft_model.named_parameters():

    if "lora_B" in name and param.requires_grad:

        lora_param_name = name

        # detach：脱离自动求导计算图
        # clone：复制数据，保存独立快照
        lora_param_before = param.detach().clone()

        break

if lora_param_name is None:
    raise RuntimeError("No trainable LoRA B parameter found.")

print(f"Tracked parameter: {lora_param_name}")


# ============================================================
# 9. 开始 LoRA 训练
# ============================================================

peft_model.train()

global_step = 0
loss_history = []

for epoch in range(NUM_EPOCHS):

    print(f"\n===== Epoch {epoch + 1}/{NUM_EPOCHS} =====")

    for batch in train_dataset:

        global_step += 1

        # 9.1 Forward
        outputs = peft_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        # 9.2 Loss
        loss = outputs.loss

        # 9.3 Backward：计算梯度
        loss.backward()

        # 9.4 Optimizer：更新 LoRA 参数
        optimizer.step()

        # 9.5 清空梯度
        optimizer.zero_grad()

        # 9.6 记录 Loss
        loss_value = loss.item()

        loss_history.append(loss_value)

        print(
            f"Step {global_step:02d} | "
            f"Loss: {loss_value:.4f}"
        )


# ============================================================
# 10. 检查 LoRA 参数是否发生变化
# ============================================================

lora_param_after = dict(
    peft_model.named_parameters()
)[lora_param_name].detach()

max_diff = (
    lora_param_after - lora_param_before
).abs().max().item()

lora_parameter_changed = max_diff > 0


# ============================================================
# 11. 输出训练总结
# ============================================================

print("\n===== Training Summary =====")

print(f"Total steps: {global_step}")

print(f"First loss: {loss_history[0]:.4f}")
print(f"Last loss: {loss_history[-1]:.4f}")

print(f"Tracked parameter: {lora_param_name}")
print(f"Max parameter change: {max_diff:.10f}")

print(
    "LoRA parameter changed:",
    lora_parameter_changed,
)


# ============================================================
# 12. 保存 LoRA Adapter
# ============================================================

ADAPTER_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

peft_model.save_pretrained(ADAPTER_DIR)

print(f"\nAdapter saved to: {ADAPTER_DIR}")


# ============================================================
# 13. 保存训练日志
# ============================================================

training_log = {
    "model_id": MODEL_ID,

    "lora_config": {
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },

    "training_config": {
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "num_examples": len(train_dataset),
        "batch_size": 1,
        "optimizer": "AdamW",
        "seed": 42,
    },

    "training_results": {
        "total_steps": global_step,
        "loss_history": loss_history,
        "first_loss": loss_history[0],
        "last_loss": loss_history[-1],
    },

    "parameter_check": {
        "tracked_parameter": lora_param_name,
        "max_parameter_change": max_diff,
        "lora_parameter_changed": lora_parameter_changed,
    },
}

with open(
    LOG_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        training_log,
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"Training log saved to: {LOG_PATH}")


# ============================================================
# 14. 检查保存结果
# ============================================================

print("\n===== Saved Files =====")

for file_path in sorted(OUTPUT_DIR.rglob("*")):

    if file_path.is_file():
        print(file_path.relative_to(DAY_DIR))

print("\nTraining completed successfully.")
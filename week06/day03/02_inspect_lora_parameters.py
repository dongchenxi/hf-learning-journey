from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="float32",
)

# 2. 创建 LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# 3. 添加 LoRA Adapter
peft_model = get_peft_model(model, lora_config)


# 4. 检查第一个 Transformer Block 的 q_proj 参数
print("\n=== Layer 0 q_proj Parameters ===")

for name, param in peft_model.named_parameters():
    if (
            "layers.0.self_attn.q_proj" in name
            or "layers.0.self_attn.v_proj" in name
    ):        print(
            f"name={name}\n"
            f"shape={tuple(param.shape)}\n"
            f"requires_grad={param.requires_grad}\n"
        )


# 5. 统计可训练参数
trainable_params = 0
frozen_params = 0

for name, param in peft_model.named_parameters():

    if param.requires_grad:
        trainable_params += param.numel()
    else:
        frozen_params += param.numel()

print("\n=== Parameter Summary ===")
print(f"Trainable params: {trainable_params:,}")
print(f"Frozen params:    {frozen_params:,}")
print(f"Total params:     {trainable_params + frozen_params:,}")
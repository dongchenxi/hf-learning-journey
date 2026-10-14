from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


# 1. 加载原始模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="float32",
)

# 2. 定义 LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# 3. 创建 LoRA Adapter
peft_model = get_peft_model(
    model,
    lora_config,
)

# 4. 打印可训练参数
print("\n=== Trainable Parameters ===")
peft_model.print_trainable_parameters()
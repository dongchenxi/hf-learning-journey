from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# 1. 配置
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

DAY_DIR = Path(__file__).resolve().parent

ADAPTER_DIR = (
    DAY_DIR.parent
    / "day04"
    / "outputs"
    / "lora_adapter"
)


# ============================================================
# 2. 检查 Adapter 文件
# ============================================================

print("Adapter directory:", ADAPTER_DIR)

if not ADAPTER_DIR.exists():
    raise FileNotFoundError(
        f"Adapter directory not found: {ADAPTER_DIR}"
    )


# ============================================================
# 3. 加载 Base Model
# ============================================================

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
)

print("Base Model loaded.")


# ============================================================
# 4. 加载 LoRA Adapter
# ============================================================

lora_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
)

print("LoRA Adapter loaded.")


# ============================================================
# 5. 切换推理模式
# ============================================================

lora_model.eval()

print("LoRA Model ready for inference.")


# ============================================================
# 6. 查看模型类型
# ============================================================

print("\n===== Model Information =====")

print("Model type:", type(lora_model).__name__)

print(
    "Base Model:",
    lora_model.peft_config["default"].base_model_name_or_path,
)

print(
    "Adapter type:",
    lora_model.peft_config["default"].peft_type,
)

print("\nAdapter loading completed successfully.")
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# 1. 基础配置
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

DAY_DIR = Path(__file__).resolve().parent

ADAPTER_DIR = (
    DAY_DIR.parent
    / "day04"
    / "outputs"
    / "lora_adapter"
)

PROMPT = "如何申请退款？"

MAX_NEW_TOKENS = 80


# ============================================================
# 2. 加载 Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Tokenizer loaded.")


# ============================================================
# 3. 准备相同的 Prompt
# ============================================================

messages = [
    {
        "role": "user",
        "content": PROMPT,
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    text,
    return_tensors="pt",
    add_special_tokens=False,
)

print("\n===== Prompt =====")
print(PROMPT)


# ============================================================
# 4. 定义推理函数
# ============================================================

def generate_answer(model, inputs):

    model.eval()

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 只提取新生成的 Token
    input_length = inputs["input_ids"].shape[1]

    generated_ids = outputs[0][input_length:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return answer


# ============================================================
# 5. Base Model 推理
# ============================================================

print("\n===== Loading Base Model =====")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
)

base_answer = generate_answer(
    base_model,
    inputs,
)

print("\n===== Base Model Answer =====")
print(base_answer)


# ============================================================
# 6. 释放 Base Model
# ============================================================

del base_model


# ============================================================
# 7. LoRA Model 推理
# ============================================================

print("\n===== Loading LoRA Model =====")

lora_base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
)

lora_model = PeftModel.from_pretrained(
    lora_base_model,
    ADAPTER_DIR,
)

lora_answer = generate_answer(
    lora_model,
    inputs,
)

print("\n===== LoRA Model Answer =====")
print(lora_answer)


# ============================================================
# 8. 对比结果
# ============================================================

print("\n===== Comparison =====")

print("Prompt:", PROMPT)

print("\nBase Model:")
print(base_answer)

print("\nLoRA Model:")
print(lora_answer)

print("\nAnswers identical:", base_answer == lora_answer)

print("\nComparison completed.")
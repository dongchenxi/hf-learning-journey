import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# 1. 固定模型
model_name = "Qwen/Qwen2.5-0.5B"


# 2. 固定测试 Prompt
prompt = "我的订单什么时候发货？"


# 3. 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 4. 加载 Base Model
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()


# 5. Tokenization
inputs = tokenizer(
    prompt,
    return_tensors="pt"
)


# 6. 固定生成参数
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )


# 7. 只提取新生成的 Token
#获取输入 Prompt 的 Token 数量。
input_length = inputs["input_ids"].shape[1]
#第二行的作用：从输入 Prompt 结束的位置开始切片，只保留模型新生成的 Token。
generated_ids = outputs[0][input_length:]


# 8. Decode
response = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True
)


# 9. 输出结果
print("=== Model ===")
print(model_name)

print("\n=== Prompt ===")
print(prompt)

print("\n=== Base Model Response ===")
print(repr(response))
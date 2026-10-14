from pathlib import Path

from huggingface_hub import HfApi


# ============================================================
# 1. 配置
# ============================================================

REPO_ID = "CHENXI0312/qwen2.5-0.5b-customer-service-lora"

DAY_DIR = Path(__file__).resolve().parent

ADAPTER_DIR = (
    DAY_DIR.parent
    / "day04"
    / "outputs"
    / "lora_adapter"
)


# ============================================================
# 2. 检查本地 Adapter
# ============================================================

required_files = [
    "adapter_config.json",
    "adapter_model.safetensors",
]

for filename in required_files:
    file_path = ADAPTER_DIR / filename

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

print("Adapter files verified.")

for file_path in sorted(ADAPTER_DIR.iterdir()):
    if file_path.is_file():
        print(f"  {file_path.name}")


# ============================================================
# 3. 创建 Hugging Face API 客户端
# ============================================================

api = HfApi()


# ============================================================
# 4. 创建私有模型仓库
# ============================================================

repo_url = api.create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    private=True,
    exist_ok=True,
)

print(f"\nRepository: {repo_url}")


# ============================================================
# 5. 上传 LoRA Adapter
# ============================================================

upload_result = api.upload_folder(
    folder_path=str(ADAPTER_DIR),
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload Week 6 Day 4 LoRA adapter",
)

print("\nAdapter uploaded successfully.")
print(f"Commit URL: {upload_result}")
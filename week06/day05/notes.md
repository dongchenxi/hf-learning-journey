# Week 6 Day 5：重新加载、比较并上传 LoRA Adapter

## 1. 学习目标与实际配置

本日完成 Day 4 训练产物的推理复用与远程发布闭环：重新加载本地 LoRA Adapter、在受控条件下比较 Base Model 与 LoRA Model 输出，并将 Adapter 上传到 Hugging Face Hub 后验证远程状态。

| 项目 | 实际值 |
| --- | --- |
| 基础模型 | `Qwen/Qwen2.5-0.5B-Instruct` |
| 本地 Adapter | `week06/day04/outputs/lora_adapter/` |
| Adapter 加载方式 | `PeftModel.from_pretrained(base_model, ADAPTER_DIR)` |
| 对比 Prompt | `如何申请退款？` |
| Chat 模板 | 同一个 `tokenizer.apply_chat_template(..., add_generation_prompt=True)` |
| 生成设置 | `do_sample=False`、`max_new_tokens=80` |
| Hub 仓库 | `CHENXI0312/qwen2.5-0.5b-customer-service-lora` |
| 本次远程提交 | `12772d562f1b8defd4f5ada260e8762008e740cf` |

## 2. 重新加载 LoRA Adapter

`01_load_lora_adapter.py` 先加载基础模型，再将 Day 4 保存的 Adapter 附加到它：

```python
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
lora_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
lora_model.eval()
```

脚本成功加载 `Qwen/Qwen2.5-0.5B-Instruct` 与本地 Adapter，并进入推理模式。Adapter 并不包含完整基础模型权重，因此重新推理需要“基础模型 + Adapter”。若本地 Adapter 已删除，只要 Hugging Face Hub 仓库仍存在且拥有访问权限，也可以将 Repo ID `CHENXI0312/qwen2.5-0.5b-customer-service-lora` 作为 Adapter 来源重新加载。

## 3. Base Model 与 LoRA Model 的受控比较

`02_compare_base_lora.py` 对两个模型使用完全相同的输入构造、chat template 与确定性生成参数；输出结果为：

```text
Prompt: 如何申请退款？
Answers identical: False
```

实际观察中，Base Model 给出联系客服等较泛化的建议，但回答在 `max_new_tokens=80` 限制下截断；LoRA Model 的语气更礼貌，但没有给出明确的退款步骤。

因此，这个实验只能证明加载 Adapter 后的输出与基础模型不同，不能证明 LoRA Model 的质量、退款任务表现或泛化能力有所提升。结论受以下限制：

- 仅使用一个 Prompt；
- 没有独立测试集、人工评分或任务指标；
- 输出受 80 个新 token 的截断影响；
- 单次对比不能代表其他问题类型或真实用户场景。

后续若要评估效果，应使用未参与训练的多样化样本，并事先定义可复现的评价标准。

## 4. 上传与远程验证

`03_upload_lora.py` 在上传前检查本地 `adapter_config.json` 和 `adapter_model.safetensors`，随后使用 `HfApi.create_repo(..., private=True, exist_ok=True)` 创建或访问私有模型仓库，并通过 `upload_folder` 上传 Adapter 目录。

上传成功后，使用 `HfApi.repo_info` 与 `HfApi.list_repo_files` 进行独立验证，实际结果为：

```text
Repo ID: CHENXI0312/qwen2.5-0.5b-customer-service-lora
Private: True
Remote Files:
.gitattributes
README.md
adapter_config.json
adapter_model.safetensors
```

上传成功的提示不能替代验证：远程文件检查确认必要的配置与权重都存在；`Private: True` 则确认仓库访问范围符合预期。该仓库为私有仓库，后续从 Hub 加载时仍须使用具有访问权限的 Hugging Face 身份或令牌。

## 5. Git 与产物边界

Day 4 的本地 Adapter 目录属于可重新生成的模型产物，已由 `.gitignore` 排除，不提交权重文件。本日提交保留加载、比较与上传脚本，以及本实验笔记；`peft` 作为 LoRA Adapter 加载所需依赖，与对应的 `uv.lock` 更新一并版本控制。

## 6. 本日结论

1. LoRA Adapter 可以附加到对应的基础模型后用于推理；本地文件丢失时，可在有权限的前提下从 Hub Repo ID 重新加载。
2. 公平比较必须保持 Prompt、chat template 和生成参数一致；“输出不同”不等于“模型更好”。
3. 上传后要分别验证远程文件完整性和私有可见性。

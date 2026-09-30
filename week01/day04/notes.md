# Week 1 Day 4 - LLM Inference Parameters
# 第一周第四天 - LLM 推理参数

## Core Memory / 核心记忆

- **max_new_tokens = output length = 控制最多生成多少个新 token**
- **temperature = randomness = 控制采样随机程度**
- **top_p = cumulative probability range = 根据累计概率控制候选 token 范围**
- - **top_k = number of candidates = 控制保留多少个候选 token**


---

## max_new_tokens

`max_new_tokens` controls the maximum number of new tokens generated after the prompt.

`max_new_tokens` 控制模型在 Prompt 后最多生成多少个新的 token。

---

## Temperature

Temperature changes how concentrated or spread out the token probability distribution is during sampling.

Temperature 会影响采样时 token 概率分布的集中或分散程度。

- **Lower temperature = more stable and conservative = 更稳定、更保守**
- **Higher temperature = more random and diverse = 更随机、更多样**
- **Higher temperature does not mean better quality = temperature 越高不代表质量越好**

---

## top_p

`top_p` keeps a dynamic set of candidate tokens based on cumulative probability.

`top_p` 根据累计概率动态决定候选 token 范围。

- **Smaller top_p = narrower candidate range = 候选范围更窄**
- **Larger top_p = wider candidate range = 候选范围更大**

`top_p = 0.8` does NOT mean keeping 80% of all tokens.

`top_p = 0.8` 不是保留全部 token 的 80%，而是从高概率 token 开始累加，直到累计概率达到约 0.8。

## Text Generation Flow / 文本生成流程

Text
→ Tokenizer
→ Token IDs
→ GPT-2
→ generate()
→ Generated Token IDs(原始 Prompt 的 Token IDs +新生成的 Token IDs)
→ Tokenizer.decode()
→ Text

- **AutoTokenizer = Text ↔ Token IDs**
- **AutoModelForCausalLM = 根据前面的 token 预测下一个 token**
- **max_new_tokens = 最多生成多少个新 token**
- **do_sample=False = 关闭随机采样，方便控制变量**
- **decode() = Token IDs → Text**

## Experiment Results / 实验结果

### max_new_tokens-控制最多生成多少个新的token

- 10:
- 30:
- 60:

My observation / 我的观察：

---

### temperature-控制采样随机程度

- 0.2:
- 0.7:
- 1.2:

My observation / 我的观察：

---

### top_p累计概率控制候选概率

- 0.5:
- 0.8:
- 0.95:

My observation / 我的观察：
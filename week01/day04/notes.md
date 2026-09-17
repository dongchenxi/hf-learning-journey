## Experiment Conclusions / 实验结论

### Temperature Experiment / Temperature 实验

#### temperature = 0.2

The output is more conservative and stable.

输出更保守、更稳定，但内容可能更重复。

#### temperature = 0.7

The output becomes more diverse and flexible.

输出开始更多样、更灵活，同时仍然保持一定稳定性。

#### temperature = 1.2

The output becomes more random and less stable.

输出更随机、更发散，也更容易出现重复、逻辑跳跃或奇怪表达。

### Core Memory / 核心记忆

- **Lower temperature = more stable and conservative = 更稳定、更保守**
- **Higher temperature = more random and diverse = 更随机、更多样**
- **Higher temperature does not mean better quality = temperature 越高不代表质量越好**

---

### top_k Experiment / top_k 实验

#### top_k = 5

The candidate set is small, so the output is usually more conservative and focused.

候选 token 数量较少，因此输出通常更保守、更集中，也可能更容易重复。

#### top_k = 20

The model has more candidate tokens to choose from, so the output becomes more flexible.

模型有更多候选 token 可以选择，因此输出会更灵活、更多样。

#### top_k = 50

The candidate set becomes larger, which can increase diversity but may also reduce stability.

候选范围进一步扩大，输出会更多样，但也可能更发散、更不稳定。

### Core Memory / 核心记忆

- **Smaller top_k = fewer candidates = 候选更少，更保守**
- **Larger top_k = more candidates = 候选更多，输出更多样**
- **top_k = fixed number of candidate tokens = top_k 控制固定数量的候选 token**

---

### top_p Experiment / top_p 实验

#### top_p = 0.5

The candidate probability range is smaller, so the output is usually more conservative and may become repetitive.

候选概率范围较小，因此输出通常更保守，也可能更容易重复。

#### top_p = 0.8

The output becomes more balanced and diverse.

输出通常更平衡，也更多样。

#### top_p = 0.95

The candidate range becomes larger, which can increase diversity but may also make the output more unpredictable.

候选范围更大，输出更多样，但也可能更发散、更不稳定。

### Core Memory / 核心记忆

- **Smaller top_p = narrower candidate range = 候选范围更窄，更保守**
- **Larger top_p = wider candidate range = 候选范围更大，输出更多样**
- **top_p = cumulative probability = top_p 根据累计概率动态决定候选 token 范围**

---

## Final Comparison / 最终对比

- **temperature = controls randomness = 控制随机程度**
- **top_k = controls how many candidate tokens remain = 控制保留多少个候选 token**
- **top_p = controls cumulative probability range = 控制保留多大的累计概率范围**

### One-line Memory / 一句话记忆

- **temperature：有多随机**
- **top_k：留多少个候选**
- **top_p：留多大的概率范围**
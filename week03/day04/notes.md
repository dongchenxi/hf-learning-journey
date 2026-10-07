1. Dataset Tokenization 的核心是 dataset.map(tokenize_function, batched=True)。

2. map() 负责应用处理函数，tokenize_function() 定义处理逻辑，Tokenizer 负责把 text 转换成 input_ids 和 attention_mask。

3. batched=True 表示 map() 一次把一批 examples 传给函数，此时 batch["text"] 是多个文本。

4. batched=True 和 padding=True 不是一回事：前者控制 map() 如何提供数据，后者控制 Tokenizer 如何补齐序列。

5. Dataset Tokenization 阶段通常先保存可变长度的 token 序列，不需要立即 return_tensors="pt"；真正组成模型训练 batch 时再完成 Padding/Tensor 化。
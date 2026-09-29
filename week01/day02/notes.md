## Core Memory / 核心记忆

- **pipeline() = preprocessing + model inference + postprocessing**
- **pipeline() = 预处理 + 模型推理 + 后处理**
Raw Text / 原始文本
        ↓
Preprocessing / 预处理
        ↓
Model Inference / 模型推理
        ↓
Postprocessing / 后处理
        ↓
Readable Result / 可读结果
- 
- **task = what to do = 做什么**
- **model = which model to use = 用哪个模型**
- **Model ID = Hugging Face Hub 上模型仓库的标识**

- **Text Classification = 给文本分类**
- **NER = 找出文本中的实体**
- **Text Generation = 根据已有文本继续生成**

- **pipeline ≠ model**

## Interview Practice / 双语面试练习

### Q1. What is a pipeline in Hugging Face?
### Hugging Face 中的 pipeline 是什么？

**English**

A pipeline is a high-level inference API that combines preprocessing, model inference, and postprocessing.

**中文**

pipeline 是一个高级推理接口，它把预处理、模型推理和后处理封装在一起。

---

### Q2. What does `task` mean in `pipeline()`?
### `pipeline()` 中的 task 是什么意思？

**English**

The task tells the pipeline what kind of job to perform.

**中文**

task 表示希望 pipeline 执行什么类型的任务。

---

### Q3. What is the difference between task and model?
### task 和 model 有什么区别？

**English**

The task defines what to do, while the model defines which model is used to perform the task.

**中文**

task 决定“做什么”，model 决定“用哪个模型来做”。

---

### Q4. What is NER?
### 什么是 NER？

**English**

NER stands for Named Entity Recognition. It identifies entities such as people, organizations, and locations in text.

**中文**

NER 是 Named Entity Recognition，即命名实体识别，用来识别文本中的人名、组织、地点等实体。

---

### Q5. Why is the first model run usually slower?
### 为什么第一次运行模型通常比较慢？

**English**

Because the model files and tokenizer may need to be downloaded from Hugging Face Hub and cached locally.

**中文**

因为第一次运行时通常需要从 Hugging Face Hub 下载模型权重、Tokenizer 等文件，并缓存到本地。
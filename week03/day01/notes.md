datasets
Hugging Face 的一个 Python library。

Dataset 
一个数据 split

load_dataset()
从数据源加载 Dataset

DatasetDict
包含 train / validation / test 等多个 Dataset split

split 
Dataset 的数据分区

features
→ 有哪些列 + 每列的数据类型/结构

=========自测问题==========

## Week 3 · Day 1 自测问题

1. **`datasets` 是什么？`load_dataset()` 是做什么的？**
datasets是hugging face的一个python library
load_dataset()用于加载dataset的函数
2. **`Dataset`、`DatasetDict`、split 三者是什么关系？**
DatasetDict
│
├── train       ← split
│    └── Dataset
├── validation  ← split
│    └── Dataset
└── test        ← split
     └── Dataset
DatasetDict用于组织多个Dataset split，train，validation，test是split，每个split有Dataset

3. **如何访问 Dataset 中的 split、row 和 column？分别返回什么？**
dataset["train"]          # split → Dataset
dataset["train"][0]       # row → 一条 example，通常是 dict
dataset["train"]["text"]  # column → text 列的数据

4. **`column_names`、`features`、`num_rows` 分别是什么？**
column_names
→ Dataset 有哪些列

features
→ Dataset 每一列及对应的数据类型/结构

num_rows
→ Dataset 有多少行

5. **`Features` 是什么？**
Dataset 每一列的名称以及对应的数据类型/结构

6. **为什么 Dataset 中的 `label` 可能是 `0`、`1`，而不是 `"neg"`、`"pos"`？如何得到类别名称？**
label 使用 0、1 这样的整数作为类别 ID
类别 ID和类别名称对应关系保存在ClassLabel，使用int2str，将类别ID转变为类别名称
7. **`load_dataset(...)` 和 `load_dataset(..., split="train")` 有什么区别？**

8. **Week 3 的 Dataset 学习与 Week 2 的 Sequence Classification 推理流程是什么关系？** :chatgpt-content-reference{index="0"}
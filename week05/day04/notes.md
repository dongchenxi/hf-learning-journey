# Week 5 Day 4：Instruction Data

## 一、学习目标

将普通客服 QA 数据转换为 Instruction Dataset 和 Messages Dataset，
保存为 JSONL，并验证数据结构及内容一致性。

## 二、核心知识点

### 1. 普通 QA 数据

使用 `question` 保存用户问题，使用 `answer` 保存客服回答。

### 2. Instruction Dataset

Instruction Dataset 使用 `instruction` 描述任务、
`input` 提供任务输入、`output` 提供期望回答。

这是常见的数据组织方式，并非所有指令微调数据集都必须采用这三个字段。

### 3. Messages Dataset

Messages Dataset 使用按顺序排列的消息表示对话，
每条消息通过 `role` 标识说话者，通过 `content` 保存消息内容。

本次实验采用：

- system：定义客服助手角色
- user：用户问题
- assistant：客服回答

Messages 是结构化数据，Chat Template 负责将其格式化为模型需要的输入形式。

### 4. JSONL

JSONL 文件每一行都是一个独立、有效的 JSON 值。

本次数据集中，每行是一个 JSON 对象，表示一条训练样本。

Messages Dataset 的一条样本可以包含多条消息。

### 5. JSON 序列化与解析

- `json.dumps()`：Python 对象 → JSON 字符串
- `json.loads()`：JSON 字符串 → Python 对象
- `json.dump()`：将 Python 对象序列化并写入文件

### 6. 文件路径

相对路径依赖当前工作目录。

使用：

`Path(__file__).resolve().parent / "data"`

可以将输出目录固定在当前 Python 脚本所在目录下。

## 三、实验记录

### 实验 1：单条 QA 转换

文件：`01_convert_single_qa.py`

结果：成功生成 Instruction 和 Messages 两种数据结构。

### 实验 2：批量 QA 转换

文件：`02_convert_batch_qa.py`

结果：

- Original QA count: 3
- Instruction count: 3
- Messages count: 3

### 实验 3：保存 JSONL

输出文件：

- `data/instruction.jsonl`
- `data/messages.jsonl`

验证结果：两个文件各有 3 行。

### 实验 4：读取并验证 JSONL

文件：`03_validate_jsonl.py`

实际输出：

- Instruction count: 3
- Messages count: 3
- Instruction structure: PASS
- Messages structure: PASS
- QA content consistency: PASS

验证范围：两种转换结果的字段结构、角色顺序及问答内容一致性。

## 四、核心总结

普通客服 QA 数据可以转换成 Instruction Dataset 或 Messages Dataset。

两种格式的数据结构不同，但可以表达相同的问答任务。

JSONL 适合逐行保存和读取训练样本。

保存时使用 `json.dumps()`，读取时使用 `json.loads()`。

数据转换完成后，需要验证样本数量、字段结构和内容一致性。
# Week 5 Day 2 — System / User / Assistant

## 1. Chat Messages

Chat Messages 是按顺序排列的消息列表，
每条消息通常包含 role 和 content，
用于表示消息的发送角色及其内容。

## 2. 三种角色

- system：提供助手的整体行为指导。
- user：表示用户输入。
- assistant：表示助手回复。

## 3. 对话历史

对话历史是之前的 User 和 Assistant 消息，
它们为后续回复提供上下文。

主要作用：

1. 帮助理解上下文。
2. 保持对话连续性。

## 4. 添加消息

使用 messages.append() 在消息列表末尾添加新消息。

```python
messages.append({
    "role": "assistant",
    "content": assistant_reply
})
# 蒸馏 DeepSeek-R1 知识到自定义小模型

## 数据集介绍

Magpie-Reasoning-V2 数据集，其中包含由 DeepSeek-R1 生成的 250K CoT 推理样本。这些样本涵盖了数学推理、编码和一般问题解决等不同的任务。

### 数据集结构

每个样本包括：

- 指令：任务描述（例如：“解决这道数学题”）
- 回应：DeepSeek-R1 的逐步推理(CoT)

```json
{
  "instruction": "Solve for x: 2x + 5 = 15",
  "response": "<think>First, subtract 5 from both sides: 2x = 10. Then, divide by 2: x = 5.</think>"
}
```

将数据集构造成如下的聊天模板格式：

```python
"<|im_start|>user\n"
f"{example['instruction']}\n"
"<|im_end|>\n"
"<|im_start|>assistant\n"
f"{example['response']}\n"
"<|im_end|>"
```

> 每个 LLM 使用特定的指令和任务格式。将数据集与这种结构对其可以确保模型学习到正确的回话模式。所以一定要根据你想要蒸馏的模型来格式化数据

## 环境安装

```bash
pip install torch transformers datasets accelerate bitsandbytes
```

安装 flash-attn

```bash
pip install flash-attn --no-build-isolation
```

## 参考链接

[如何蒸馏 DeepSeek-R1](https://www.cnblogs.com/little-horse/p/18701373)

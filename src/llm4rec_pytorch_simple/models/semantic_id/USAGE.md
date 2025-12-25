# SemanticIdTokenizer 使用指南

## 简介

`SemanticIdTokenizer` 用于将物品 ID 转换为语义 ID 序列，支持冲突处理和约束生成。

## 快速开始

```python
from llm4rec_pytorch_simple.models.semantic_id import SemanticIdTokenizer

# 推荐方式：直接从预构建的映射加载
tokenizer = SemanticIdTokenizer.from_pretrained(
    'ml-1m/multimodal_datasets/rqvae_results/semantic_id_mapping_text.pkl'
)

# 或者空初始化后加载
tokenizer = SemanticIdTokenizer()
tokenizer.load_mapping('semantic_id_mapping_text.pkl')
```

> **注意**：映射文件 `semantic_id_mapping_*.pkl` 由 `train_rqvae.py` 训练时自动生成

## 核心功能

### 1. Tokenize（物品 ID → 语义 ID）

```python
# 单个物品
semantic_ids = tokenizer.tokenize(1)  # item_id 为 int
# 输出: tensor([[9, 20, 3, 0]])

# 批量处理
item_ids = [1, 10, 100]
semantic_ids = tokenizer.tokenize(item_ids)
# 输出: tensor([[ 9, 20,  3,  0],
#               [24, 35, 13,  0],
#               [ 9,  3, 21,  1]])
```

### 2. Decode（语义 ID → 物品 ID）

```python
item_ids = tokenizer.decode(semantic_ids)
# 输出: [1, 10, 100]
```

### 3. 约束生成

在推理阶段，确保生成的 ID 序列是合法的：

```python
# 检查前缀是否合法
prefix = [9, 20]
is_valid = tokenizer.is_prefix(prefix)  # True

# 获取可能的下一个 token
next_tokens = tokenizer.get_valid_next_tokens(prefix)
# 输出: [3, 41, 0, 18, 53, 21, 13, 26, 29, ...]

# 检查序列是否完整
is_complete = tokenizer.is_complete_sequence([9, 20, 3, 0])  # True
```

### 4. 获取统计信息

```python
stats = tokenizer.get_stats()
print(stats)
# {
#     'total_items': 3883,
#     'num_codebooks': 3,
#     'semantic_id_length': 4,
#     'vocab_size': 86
# }

## 冲突处理

当多个物品映射到相同的原始语义 codes 时，会自动添加**消歧索引**：

```python
# 物品 1:   [9, 20, 3] → [9, 20, 3, 0]
# 物品 122: [9, 20, 3] → [9, 20, 3, 1]
# 物品 220: [9, 20, 3] → [9, 20, 3, 2]
```

统计信息：

- **总物品数**: 3883
- **冲突序列数**: 240 (原始 3 维 codes 有 240 个重复)
- **最大冲突数**: 86 (同一序列最多对应 86 个物品)
- **扩展后长度**: 4 (原始 3 + 消歧索引 1)

## 在训练中使用

### Decoder 训练时

```python
# 将历史序列中的物品 ID 转换为语义 ID
history_item_ids = [10, 20, 30]  # 用户历史（item_id 为 int）
semantic_ids = tokenizer.tokenize(history_item_ids)
# semantic_ids: (3, 4)

# 作为 Decoder 输入
# 展平或根据模型需求调整形状
```

### 推理时的约束生成

```python
# 生成过程中，每个位置都验证前缀
current_prefix = [9, 20]

# 获取合法的下一个 token
valid_tokens = tokenizer.get_valid_next_tokens(current_prefix)

# 在 logits 上应用约束（mask）
logits[~valid_tokens] = -float('inf')  # mask 掉非法 token
```

## 完整工作流程

### 1. 训练 RQVAE 并生成映射

```bash
python -m llm4rec_pytorch_simple.models.semantic_id.rqvae_module.train_rqvae \
    --mode text \
    --epochs 2000
```

这会生成两个文件：

- `rqvae_codes_text.parquet`：codes 数据
- `semantic_id_mapping_text.pkl`：完整映射（推荐使用）

### 2. 加载并使用 Tokenizer

```python
from llm4rec_pytorch_simple.models.semantic_id import SemanticIdTokenizer
import torch

# 初始化
tokenizer = SemanticIdTokenizer.from_pretrained(
    'ml-1m/multimodal_datasets/rqvae_results/semantic_id_mapping_text.pkl'
)

# 训练数据准备
user_history = [1, 10, 100, 200]
semantic_ids = tokenizer.tokenize(user_history)
print(f"Semantic IDs: {semantic_ids}")

# 验证可逆性
decoded = tokenizer.decode(semantic_ids)
print(f"Decoded: {decoded}")
assert decoded == user_history

# 推理时的约束生成
prefix = [9, 20]
if tokenizer.is_prefix(prefix):
    next_tokens = tokenizer.get_valid_next_tokens(prefix)
    print(f"Valid next tokens: {next_tokens[:10]}")  # 显示前 10 个
```

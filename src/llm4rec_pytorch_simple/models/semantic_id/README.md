# Semantic ID 模块说明

## 概述

Semantic ID 是生成式推荐系统中的核心概念，用于将 Item ID 转换为语义化的 token 序列，使得推荐任务可以像语言模型生成文本一样生成推荐结果。

## 为什么需要 Semantic ID？

### 传统方法的局限

在传统的 Embedding-based 推荐系统中：

```python
# 传统方法
Item 123 → Dense Embedding [0.1, -0.3, 0.5, 0.8, ...]  # 维度通常是 64-512
```

**问题**：

1. **计算复杂度高**：推理时需要计算用户表示与所有 item embeddings 的相似度 `O(N × D)`
2. **可扩展性差**：item 数量增加时，计算量线性增长
3. **缺乏语义结构**：Item ID 只是数字标识，没有语义信息
4. **冷启动困难**：新 item 需要收集足够交互才能学到好的 embedding

### Semantic ID 的优势

```python
# Semantic ID 方法
Item 123 → Semantic ID [45, 128, 67, 9]  # 长度固定，每个位置是 0-255 的整数
```

**优势**：

1. **计算复杂度低**：生成 K 个推荐只需 `O(K × L × V)`，其中 L 是 ID 长度，V 是 codebook 大小
2. **可扩展性强**：与 item 总数无关，可以轻松扩展到百万、千万级别
3. **语义聚类**：相似的 item 可以有相似的 Semantic ID 前缀
4. **支持冷启动**：可以基于 item 内容特征生成 Semantic ID

## Semantic ID 的工作原理

### 1. ID 结构

Semantic ID 是一个固定长度的 token 序列：

```
Semantic ID = [token_0, token_1, token_2, token_3]
```

每个 token 的取值范围是 `[0, codebook_size-1]`，通常 `codebook_size = 256`。

**容量计算**：

- `id_length=4, codebook_size=256`: 可表示 256^4 = 4,294,967,296 个不同的 item
- `id_length=3, codebook_size=256`: 可表示 256^3 = 16,777,216 个不同的 item

### 2. 层次化语义

理想情况下，Semantic ID 的每个位置编码不同粒度的语义信息：

```
Item: 《泰坦尼克号》
Semantic ID: [45, 128, 67, 9]
             ↓   ↓    ↓   ↓
           类型 年代  评分 具体ID

位置 0 (45):  爱情片
位置 1 (128): 1990年代
位置 2 (67):  高评分 (8-9分)
位置 3 (9):   具体的电影标识
```

这样，相似的电影会有相似的 Semantic ID 前缀：

```
《泰坦尼克号》  [45, 128, 67, 9]   # 爱情/1990s/高分
《诺丁山》      [45, 128, 71, 3]   # 爱情/1990s/高分
                ^^  ^^^            # 前两位相同！

《终结者》      [201, 55, 12, 88]  # 科幻/1980s/动作
                ^^^                # 完全不同的前缀
```

### 3. 映射表

由于 Semantic ID 和 Item ID 是两个不同的标识系统，需要维护双向映射：

```python
# Item ID → Semantic ID
item_to_semantic = {
    123: [45, 128, 67, 9],
    124: [45, 128, 71, 3],
    ...
}

# Semantic ID → Item ID
semantic_to_item = {
    (45, 128, 67, 9): 123,
    (45, 128, 71, 3): 124,
    ...
}
```

## Semantic ID 的生成方法

### 方法 1：简单映射（进制转换）

最简单的方式是将 Item ID 转换为指定进制的表示：

```python
def item_id_to_semantic_id(item_id, id_length=4, base=256):
    semantic_id = []
    for _ in range(id_length):
        semantic_id.append(item_id % base)
        item_id //= base
    return semantic_id

# 示例
item_id_to_semantic_id(0)    # → [0, 0, 0, 0]
item_id_to_semantic_id(1)    # → [1, 0, 0, 0]
item_id_to_semantic_id(255)  # → [255, 0, 0, 0]
item_id_to_semantic_id(256)  # → [0, 1, 0, 0]
item_id_to_semantic_id(520)  # → [8, 2, 0, 0]  # 520 = 8 + 2*256
```

**优点**：

- 实现简单
- 双向转换容易
- 适合快速测试

**缺点**：

- 没有语义信息
- 相邻 ID 不一定相似
- 无法利用 item 内容特征

### 方法 2：RQ-VAE（推荐）

使用 Residual Quantized Variational Autoencoder 基于 item 特征生成语义化的 Semantic ID：

```python
# 1. 准备 item 特征
item_features = {
    123: {
        "genre": [1, 0, 0, ...],      # one-hot: 爱情片
        "year": 1997,
        "rating": 8.5,
        "tags": ["romance", "drama"],
        "embedding": [0.5, -0.3, ...]  # 预训练的 embedding
    }
}

# 2. 训练 RQ-VAE
rq_vae = RQVAE(
    feature_dim=...,
    embedding_dim=128,
    num_codebooks=4,      # Semantic ID 长度
    codebook_size=256,
)

# 3. 编码为 Semantic ID
for item_id, features in item_features.items():
    semantic_id = rq_vae.encode(features)
    # semantic_id = [45, 128, 67, 9]
```

**RQ-VAE 工作原理**：

```
Item Features → Encoder → Embedding
                             ↓
                    Quantizer 1 (粗粒度)
                             ↓ Code 1: 45 (类型)
                    Quantizer 2 (中粒度)
                             ↓ Code 2: 128 (年代)
                    Quantizer 3 (细粒度)
                             ↓ Code 3: 67 (评分)
                    Quantizer 4 (最细粒度)
                             ↓ Code 4: 9 (具体 item)

Semantic ID: [45, 128, 67, 9]
```

**优点**：

- 语义化：相似 item 有相似 ID
- 层次化：每个位置编码不同粒度
- 支持冷启动：新 item 可基于特征生成 ID
- 可解释性：ID 的每个位置有明确含义

**缺点**：

- 需要 item 特征数据
- 需要预训练 RQ-VAE
- 实现复杂度较高

## 在推荐系统中的应用

### 训练阶段

```python
# 1. 获取训练数据
user_history = [101, 205, 387]  # 用户历史 item IDs
target_item = 520                # 目标 item ID

# 2. 转换为 Semantic ID
target_semantic_id = semantic_id_manager.get_semantic_id(520)
# → [45, 128, 67, 9]

# 3. 训练模型生成 Semantic ID（类似语言模型）
# Encoder: 编码用户历史
context = encoder(user_history)

# Decoder: 自回归生成 Semantic ID
# 输入: [BOS, 45, 128, 67]
# 目标: [45, 128, 67, 9]
logits = decoder(context, input_ids=[BOS, 45, 128, 67])

# 4. 计算损失
loss = CrossEntropy(logits, target=[45, 128, 67, 9])
```

### 推理阶段

```python
# 1. 编码用户历史
user_history = [101, 205, 387]
context = encoder(user_history)

# 2. 自回归生成 Semantic ID（类似文本生成）
generated_semantic_id = []
current_input = [BOS]

for step in range(id_length):
    # 预测下一个 token
    logits = decoder(context, current_input)
    next_token = logits.argmax()
    
    generated_semantic_id.append(next_token)
    current_input.append(next_token)

# generated_semantic_id = [45, 128, 67, 9]

# 3. 映射回 Item ID
item_id = semantic_id_manager.get_item_id([45, 128, 67, 9])
# → 520

# 4. 推荐给用户
recommend(user, item_id=520)
```

### 生成 Top-K 推荐

使用 Beam Search 生成多个候选：

```python
# Beam Search 生成 Top-10
beam_size = 10
candidates = beam_search(
    encoder_output=context,
    beam_size=beam_size,
    max_length=id_length,
)

# candidates = [
#     ([45, 128, 67, 9], score=0.95),   # 最高分
#     ([45, 128, 71, 3], score=0.87),
#     ([201, 55, 12, 88], score=0.76),
#     ...
# ]

# 映射回 Item IDs
top_k_items = []
for semantic_id, score in candidates:
    item_id = semantic_id_manager.get_item_id(semantic_id)
    top_k_items.append((item_id, score))

# top_k_items = [(520, 0.95), (521, 0.87), (100, 0.76), ...]
```

## 与传统方法的对比

### Embedding-based 方法

```python
# 训练
user_repr = encoder(user_history)              # [B, D]
target_emb = embedding_layer(target_item)      # [D]
loss = bce_loss(user_repr, target_emb, negatives)

# 推理
scores = user_repr @ all_item_embeddings.T     # [B, N]
top_k = scores.topk(k)                         # O(N × D)
```

### Semantic ID 方法

```python
# 训练
context = encoder(user_history)                # [B, Seq, D]
target_sid = get_semantic_id(target_item)      # [id_length]
logits = decoder(context, target_sid)          # [B, id_length, V]
loss = cross_entropy(logits, target_sid)

# 推理
generated_sid = beam_search(context, k)        # [B, K, id_length]
top_k = map_to_items(generated_sid)            # O(K × L × V)
```

### 复杂度对比

| 操作 | Embedding-based | Semantic ID |
|------|----------------|-------------|
| **训练** | O(B × D × S) | O(B × L × V) |
| **推理** | O(N × D) | O(K × L × V) |
| **内存** | O(N × D) | O(V × D) |

其中：

- N: item 总数（通常 10^4 - 10^7）
- D: embedding 维度（通常 64-512）
- L: Semantic ID 长度（通常 3-5）
- V: codebook 大小（通常 256）
- K: Top-K 数量（通常 10-100）
- B: batch size
- S: 负采样数量

**关键优势**：推理复杂度从 O(N×D) 降低到 O(K×L×V)，当 N 很大时优势明显！

## 实现细节

### SemanticIDManager 类

负责管理 Item ID 和 Semantic ID 的双向映射：

```python
manager = SemanticIDManager(
    id_length=4,
    codebook_size=256,
    num_items=3706,
)

# 获取 Semantic ID
semantic_id = manager.get_semantic_id(520)  # → [45, 128, 67, 9]

# 反向查询
item_id = manager.get_item_id([45, 128, 67, 9])  # → 520

# 批量处理
item_ids = torch.tensor([520, 521, 522])
semantic_ids = manager.get_batch_semantic_ids(item_ids)
# → [[45, 128, 67, 9], [45, 128, 71, 3], ...]

# 保存/加载映射表
manager.save_to_file("data/semantic_id_mapping.json")
manager.load_from_file("data/semantic_id_mapping.json")
```

### 映射表格式

```json
{
  "item_to_semantic": {
    "0": [0, 0, 0, 0],
    "520": [45, 128, 67, 9],
    "521": [45, 128, 71, 3]
  },
  "semantic_to_item": {
    "[0, 0, 0, 0]": 0,
    "[45, 128, 67, 9]": 520,
    "[45, 128, 71, 3]": 521
  },
  "metadata": {
    "id_length": 4,
    "codebook_size": 256,
    "num_items": 3706,
    "generation_method": "simple_mapping"
  }
}
```

## 使用示例

### 1. 生成映射表

```bash
# 使用简单映射
python scripts/generate_semantic_ids.py \
    --num_items 3706 \
    --id_length 4 \
    --codebook_size 256 \
    --output data/semantic_id_mapping.json
```

### 2. 在代码中使用

```python
from llm4rec_pytorch_simple.models.semantic_id import SemanticIDManager

# 创建 manager
manager = SemanticIDManager(
    id_length=4,
    codebook_size=256,
)

# 加载映射表
manager.load_from_file("data/semantic_id_mapping.json")

# 训练时使用
target_item_ids = batch["target_items"]  # [B]
target_semantic_ids = manager.get_batch_semantic_ids(target_item_ids)  # [B, 4]

# 推理时使用
generated_semantic_ids = model.generate(...)  # [B, K, 4]
for b in range(batch_size):
    for k in range(top_k):
        semantic_id = generated_semantic_ids[b, k].tolist()
        item_id = manager.get_item_id(semantic_id)
        # 推荐 item_id
```

## 优势总结

1. **可扩展性**：与 item 数量解耦，可轻松扩展到百万级
2. **效率**：推理速度快，不需要计算所有 item 的相似度
3. **语义化**：相似 item 有相似 ID，模型可以学习语义结构
4. **冷启动**：新 item 可基于特征生成 ID，无需等待交互数据
5. **统一框架**：将推荐问题转化为序列生成问题，可以利用 NLP 的技术

## 参考资料

- 论文：[Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) (NeurIPS 2023)
- 相关技术：Product Quantization, Vector Quantization, RQ-VAE
- 应用场景：大规模推荐、实时推荐、跨域推荐

##

以下是 RQ-VAE 的几种“低配平替”方案，按推荐程度排序：

1. 层级 K-Means (Hierarchical K-Means) —— 最推荐
这是最接近 RQ-VAE 效果，但实现简单得多的方案。
    - 原理：
        - 第 1 层：把所有 Item 的向量（Embedding）聚成 $K$ 类（比如 10 类）。每个 Item 获得第 1 个 Token（0-9）。
        - 第 2 层：在每一个大类内部，再分别聚成 $K$ 个子类。每个 Item 获得第 2 个 Token。
        - 递归：重复这个过程，直到达到指定的深度（ID 长度）。
    - 优点：
        - 强语义性：保证了 ID 前缀相同的物品，在向量空间里也是挨在一起的（比如 [1, 5, ...] 的物品都在同一个大簇里）。
        - 易于实现：只需要 sklearn 的 KMeans 就能跑。
        可解释性好：生成的树状结构很清晰。
        缺点：它是“贪心”的（Greedy）。第一层分错了，后面就救不回来了。而 RQ-VAE 是端到端优化的，能纠正前面的错误。

2. LSH (局部敏感哈希, Locality Sensitive Hashing)
    - 原理：用多个随机的超平面把向量空间切开。如果两个向量很近，它们大概率会落在切分的同一侧。
    - 优点：极快，计算量极小。
    - 缺点：通常生成的是二进制码（0/1），如果要生成 Codebook Size > 2 的 ID（比如 0-100），效果不如聚类好。且精度通常不如 K-Means。

3. 基于类目的规则映射 (Category-based Rule)
    - 原理：如果你有物品的类目树（比如 电子产品 -> 手机 -> 苹果），直接把类目 ID 当作 Semantic ID 的前几位。
    - 优点：完全的人工语义，最准。
    - 缺点：完全依赖人工标注的类目体系，如果类目分得不细（比如所有手机都在一个类目），区分度就不够。
    💡 核心前提：你需要有“原材料”

________

无论是 RQ-VAE、层级聚类还是 LSH，它们都不能凭空造出语义。它们都需要一个输入向量（Item Embedding）。

如果你只有 Item ID，没有文本/图片特征，怎么办？

1. 先跑一个简单的模型（如 Word2Vec / Item2Vec / 简单的 SASRec），用用户行为数据训练出一个基本的 Item Embedding。
2. 然后拿这个 Embedding 去做层级 K-Means。
3. 最后生成的 ID 就可以给 TIGER/LLM 模型用了。


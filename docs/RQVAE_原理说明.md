# RQVAE (Residual Quantized Variational AutoEncoder) 原理与步骤说明

本文档基于 `src/llm4rec_pytorch_simple/multimodal/rqvae.py` 中的代码，解释 RQVAE 的工作原理和执行步骤。

## 1. 什么是 RQVAE？

RQVAE (Residual Quantized VAE) 是对标准 VQ-VAE (Vector Quantized VAE) 的一种改进。

* **VQ-VAE**: 使用**一个**码本 (Codebook) 将连续的潜在向量映射为离散的码本索引。
* **RQVAE**: 使用**多个**码本进行**残差量化 (Residual Quantization)**。它采用“由粗到细”的策略，通过多级码本逐步逼近目标向量。

**优势**: 使用较小的码本大小（例如 256）和较少的层数（例如 3 层），就能组合出巨大的离散空间（$256^3 \approx 1.6 \times 10^7$），从而更精细地保留信息，同时保持较低的计算成本。

---

## 2. 核心原理：残差量化 (Residual Quantization)

RQVAE 的核心在于 `ResidualQuantizer` 模块。它不是一次性量化整个向量，而是分步进行：

1. **第 1 级量化**: 使用第 1 个码本找到与输入向量 $z$ 最接近的码字 $e_1$。此时残差为 $r_1 = z - e_1$。
2. **第 2 级量化**: 使用第 2 个码本找到与残差 $r_1$ 最接近的码字 $e_2$。此时残差为 $r_2 = r_1 - e_2$。
3. **...**: 重复此过程直到第 $D$ 个码本。

最终的量化向量是所有级码字的**和**：
$$ z_q = e_1 + e_2 + \dots + e_D $$

---

## 3. 详细步骤 (代码解析)

以下步骤对应代码中的 `ResidualQuantizer.forward` 方法。

### 步骤 1: 初始化

* **输入**: 编码器输出的潜在向量 $z$ (Shape: `[Batch, Dim]`)。
* **变量**:
  * `residual`: 初始化为 $z$，表示当前还需要被量化的剩余信息。
  * `z_q`: 初始化为 0，用于累加每一级的量化结果。
  * `codes`: 用于存储每一级选中的码本索引。

### 步骤 2: 循环量化 (Loop over Codebooks)

对于每一个码本 `codebook[i]` (共 `num_codebooks` 个)：

1. **计算距离 (Calculate Distances)**:
    计算当前 `residual` 与当前 `codebook[i]` 中所有向量的欧氏距离。
    $$ |x - y|^2 = |x|^2 + |y|^2 - 2xy $$
    代码对应：

    ```python
    dists = (
        torch.sum(residual**2, dim=1, keepdim=True) +
        torch.sum(codebook**2, dim=1) -
        2 * torch.matmul(residual, codebook.t())
    )
    ```

2. **寻找最近邻 (Nearest Neighbor)**:
    找到距离最小的索引。

    ```python
    indices = torch.argmin(dists, dim=1)
    codes.append(indices)
    ```

3. **获取量化向量 (Quantize)**:
    根据索引从码本中取出对应的向量 $z_{q\_i}$。

    ```python
    z_q_i = F.embedding(indices, codebook)
    ```

4. **计算承诺损失 (Commitment Loss)**:
    计算当前残差与选中码字之间的 MSE 损失。这迫使编码器输出的残差尽可能靠近码本中的向量。

    ```python
    loss += F.mse_loss(z_q_i.detach(), residual)
    ```

5. **更新状态 (Update State)**:
    * **更新残差**: 从当前残差中减去量化向量，得到下一级需要拟合的更细微的残差。

        ```python
        residual = residual - z_q_i
        ```

    * **累加量化结果**:

        ```python
        z_q = z_q + z_q_i
        ```

### 步骤 3: 直通估计 (Straight-Through Estimator)

量化操作（`argmin`）是不可导的，无法直接进行反向传播。RQVAE 使用直通估计技巧：

```python
z_q = z + (z_q - z).detach()
```

* **前向传播**: 使用量化后的 `z_q`。
* **反向传播**: `(z_q - z).detach()` 的梯度为 0，因此梯度直接从 `z_q` 流向 `z`，跳过了量化层，使得编码器可以接收到来自解码器的梯度。

---

## 4. 整体模型架构 (RQVAE Class)

`RQVAE` 类将上述量化器封装在编码器-解码器结构中：

1. **Encoder (`self.encoder`)**:
    * 将高维输入 $x$ (如 896维) 压缩到低维潜在空间 $z$ (如 128维)。
    * 结构: `Linear -> ReLU -> Linear`。

2. **Quantizer (`self.quantizer`)**:
    * 执行上述的残差量化过程。
    * 输出: 量化向量 $z_q$，离散索引 `codes`，以及 `commit_loss`。

3. **Decoder (`self.decoder`)**:
    * 将量化后的向量 $z_q$ 重构回原始空间 $\hat{x}$。
    * 结构: `Linear -> ReLU -> Linear`。

4. **总损失 (Total Loss)**:

    ```python
    total_loss = recon_loss + self.beta * commit_loss
    ```

    * `recon_loss`: 重构损失 (`MSE(x_recon, x)`)，保证重构质量。
    * `commit_loss`: 量化器的承诺损失，保证编码器输出适合量化。
    * `beta`: 权重系数，平衡两项损失。



# 码本坍塌
遇到了很严重的码本坍塌问题，尝试各种超参调整和技巧（如增加 beta，使用 warmup 策略等），但效果仍不理想。

```
[2025-12-27 14:37:06,591][__main__][INFO] - 映射表构建完成:
[2025-12-27 14:37:06,591][__main__][INFO] -   - 物品总数: 3882
[2025-12-27 14:37:06,591][__main__][INFO] -   - 冲突物品数: 3881
[2025-12-27 14:37:06,591][__main__][INFO] -   - 语义ID长度: 4
[2025-12-27 14:37:06,591][__main__][INFO] -   - Vocab 大小: 3882
```

## 原文改进

> As proposed in [40], to prevent RQ-VAE from a codebook collapse, where most of the input gets mapped to only a few codebook vectors, we use k-means clustering-based initialization for the codebook. Specifically, we apply the k-means algorithm on the first training batch and use the centroids as initialization.

主要改进

  1. K-means码本初始化 (src/llm4rec_pytorch_simple/rqvae_module/rqvae.py:19-117)

  - 实现了initialize_codebooks_kmeans()方法，在第一个训练batch时用k-means聚类初始化码本
  - 这可以防止码本坍塌（codebook collapse）问题，确保码本向量分布在数据空间中
  - 对每个codebook依次进行k-means，基于残差进行聚类

  2. 优化模型架构 (RQVAE.yaml:10-14)

  - hidden_dim: [256, 128] → [384, 256] - 更平滑的维度过渡（512→384→256→128），减少信息瓶颈
  - code_dim: 64 → 128 - 增加编码维度，保留更多信息
  - codebook_size: 32 → 64 - 增加码本大小，提供更强表达能力（64³=262144 >> 3883个电影）
  - batch_size: 512 → 256 - 减小batch size，提高梯度更新频率

  3. 优化训练策略 (RQVAE.yaml:16-21)

  - beta: 0.15 → 0.05 - 大幅降低量化损失权重，让模型先专注于学习重建
  - beta_warmup_epochs: 30 → 100 - 延长warmup期，前100个epoch量化约束从0逐渐增加到0.05
  - lr: 3e-4 → 1e-3 - 提高学习率，加快收敛

  为什么这些改进有效？

  1. K-means初始化：确保码本向量从一开始就分布在真实数据空间中，避免随机初始化导致的码本坍塌
  2. 更大的code_dim和codebook_size：提供足够的表达能力来重建512维输入
  3. 更长的beta warmup：让模型在前100个epoch专注学习重建（beta从0增长），之后再逐渐引入量化约束
  4. 更平滑的维度过渡：避免信息在编码器中损失过快

```
[2025-12-27 14:46:40,439][__main__][INFO] - 映射表构建完成:
[2025-12-27 14:46:40,439][__main__][INFO] -   - 物品总数: 3882
[2025-12-27 14:46:40,439][__main__][INFO] -   - 冲突物品数: 3335
[2025-12-27 14:46:40,439][__main__][INFO] -   - 语义ID长度: 4
[2025-12-27 14:46:40,439][__main__][INFO] -   - Vocab 大小: 236
```

---

## 进一步改进 (2025-12-27)

在上述改进基础上，进行了更多优化，最终将冲突物品数从 3335 降低到约 500 左右。

### 1. 模型架构改进

#### 1.1 模型参数初始化 (`rqvae.py`)

新增 `_init_weights` 方法，对模型参数进行规范初始化：

```python
def _init_weights(self, module):
    """初始化模型参数"""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
```

- Linear 层：权重使用均值 0、标准差 0.02 的正态分布
- LayerNorm 层：权重初始化为 1，偏置初始化为 0

#### 1.2 K-means 码本初始化时机优化 (`train_rqvae.py`)

原来的实现是在第一个 batch（epoch=0）时进行 K-means 初始化，但此时 encoder 还未经过任何训练，输出的 latent 向量质量较差。

**改进**：将 K-means 初始化延迟到 epoch=1 的第一个 batch：

```python
# 在第一个batch时使用k-means初始化码本
if epoch > 0 and first_batch:
    logger.info("使用k-means初始化码本...")
    z = model.encoder(x)
    model.quantizer.initialize_codebooks_kmeans(z, max_iters=50)
    logger.info("码本初始化完成")
    first_batch = False
```

**为什么这样更好**：
- epoch=0 时 encoder 随机初始化，输出的 latent 向量分布不合理
- 经过 epoch=0 的训练后，encoder 已经学到了一些有意义的表示
- 此时再用 K-means 初始化码本，聚类中心更能反映真实数据分布

### 2. 训练策略改进

#### 2.1 新增可配置的 Loss 权重 (`RQVAE.yaml` & `train_rqvae.py`)

```yaml
# Loss 权重
recon_weight: 1.0   # 重建损失权重
codebook_weight: 1.0  # codebook损失权重
```

总损失计算公式：
```
loss = recon_weight * recon_loss + current_beta * commit_loss + codebook_weight * codebook_loss
```

这允许独立调整三个损失项的相对重要性。

#### 2.2 Beta Warmup 策略优化

```yaml
beta: 0.25
beta_schedule: "warmup"
beta_warmup_epochs: 50
```

- `warmup` 策略：前 N 个 epoch，beta 从 0 线性增加到目标值
- 让模型先专注于学习重建，再逐渐引入量化约束

#### 2.3 可选的输入归一化

```yaml
normalize_input: false  # 可选：对输入进行归一化
```

> ⚠️这个千万不要做 norm，norm 很 sb

### 3. 统计与监控改进

#### 3.1 唯一 SID 数量统计

移除了无意义的 `vocab_size` 计算，新增 `unique_sid_count` 统计：

```python
unique_sid_count = len(sid_conflict)  # 唯一的原始 code 组合数量
```

输出示例：
```
映射表构建完成:
  - 物品总数: 3883
  - 唯一SID数量: 3359
  - 冲突物品数: 524
  - 语义ID长度: 4
```

#### 3.2 最后一轮 vs 最佳模型对比

训练结束后同时输出两组统计，方便对比：

```
===== 最后一轮模型统计 =====
  - 唯一SID数量: xxxx
  - 冲突物品数: xxx

===== 最佳模型统计 =====
已加载最佳模型 (Epoch xxx)
  - 唯一SID数量: xxxx
  - 冲突物品数: xxx
```

### 4. 最终配置参数

```yaml
model:
  batch_size: 512
  hidden_dim: [256, 128]
  code_dim: 32
  num_codebooks: 3
  codebook_size: 64  # 64^3 = 262144 >> 3883

  beta: 0.25
  beta_schedule: "warmup"
  beta_warmup_epochs: 50

  recon_weight: 1.0
  codebook_weight: 1.0
  lr: 1e-3
  min_lr: 1e-4
  epochs: 200

  val_ratio: 0.1
  normalize_input: false
```

### 5. 最终效果

```
[2025-12-27 16:32:59,361][__main__][INFO] - 映射表构建完成:
[2025-12-27 16:32:59,361][__main__][INFO] -   - 物品总数: 3883
[2025-12-27 16:32:59,361][__main__][INFO] -   - 唯一SID数量: 3359
[2025-12-27 16:32:59,361][__main__][INFO] -   - 冲突物品数: 524
[2025-12-27 16:32:59,361][__main__][INFO] -   - 语义ID长度: 4
```

**改进效果对比**：

| 版本 | 冲突物品数 | 唯一SID数量 | 冲突率 |
|------|-----------|------------|--------|
| 初始版本 | 3881 | 1 | 99.97% |
| K-means初始化后 | 3335 | ~500 | 85.9% |
| **最终优化版** | **524** | **3359** | **13.5%** |

### 6. 关键改进总结

1. **模型参数初始化**：规范的权重初始化有助于训练稳定性
2. **K-means 码本初始化时机优化**: 第一个 epoch 后做k-means，利用更好的 latent 分布,效果明显
2. **Loss 权重可配置**：允许灵活调整重建、承诺、码本三个损失的平衡
3. **Beta Warmup**：先学重建再学量化，避免早期量化约束过强
4. **统计监控**：唯一 SID 数量和最后一轮/最佳模型对比，便于调参
5. **codebook_size=64**：提供足够的表达能力（64³=262144 >> 3883），这个值越大，效果越好。（整体码本的表达能力越大，效果越好）
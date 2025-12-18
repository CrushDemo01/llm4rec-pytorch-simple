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

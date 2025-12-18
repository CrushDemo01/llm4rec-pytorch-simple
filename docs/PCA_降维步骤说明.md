# PCA 降维详细步骤说明

本文档详细说明 PCA (Principal Component Analysis) 在多模态编码器中的使用步骤。

## 使用场景

在本项目中，PCA 用于两个地方：

1. **文本编码**: BERT 输出 768 维 → PCA 降至 128 维
2. **图像编码**: ResNet18 输出 512 维 → PCA 降至 128 维

---

## 完整流程

### 阶段 1: 训练阶段 (fit)

假设我们有 N 个样本，每个样本是 D 维向量（例如 BERT 的 768 维）

#### 步骤 1: 准备数据矩阵

```
输入: X = [x1, x2, ..., xN]  # 形状: (N, D)
例如: (3883, 768) - 3883 部电影，每部 768 维 BERT 特征
```

#### 步骤 2: 中心化数据

```python
# 计算每个维度的均值
mean = X.mean(axis=0)  # 形状: (D,)

# 减去均值，使数据中心化
X_centered = X - mean  # 形状: (N, D)
```

**为什么中心化？**

- PCA 寻找方差最大的方向
- 中心化后，数据围绕原点分布，方便计算协方差

#### 步骤 3: 计算协方差矩阵

```python
# 协方差矩阵描述各维度之间的相关性
Cov = (X_centered.T @ X_centered) / (N - 1)  # 形状: (D, D)
```

#### 步骤 4: 特征值分解

```python
# 对协方差矩阵进行特征值分解
eigenvalues, eigenvectors = eig(Cov)

# eigenvalues: (D,) - 每个主成分的重要性（方差）
# eigenvectors: (D, D) - 每个主成分的方向
```

#### 步骤 5: 选择主成分

```python
# 按特征值从大到小排序
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 选择前 k 个主成分（例如 k=128）
k = 128
principal_components = eigenvectors[:, :k]  # 形状: (D, k)
```

#### 步骤 6: 保存转换矩阵

```python
# 保存以下参数供后续使用：
# - mean: 用于中心化新数据
# - principal_components: 用于投影新数据
self.mean_ = mean
self.components_ = principal_components.T  # sklearn 使用转置形式
```

---

### 阶段 2: 推理阶段 (transform)

当有新数据需要降维时：

#### 步骤 1: 中心化新数据

```python
X_new_centered = X_new - self.mean_  # 使用训练时的均值
```

#### 步骤 2: 投影到主成分空间

```python
# 将数据投影到选定的主成分上
X_reduced = X_new_centered @ self.components_.T  # 形状: (N_new, k)
```

---

## 代码示例

### 文本编码中的 PCA

```python
# 1. 训练阶段
text_encoder = TextEncoder(output_dim=128)
texts = ["movie description 1", "movie description 2", ...]

# 提取 BERT 特征
bert_features = text_encoder.extract_features(texts)  # (N, 768)

# 训练 PCA
text_encoder.pca.fit(bert_features)
# 内部步骤：
#   - 计算均值: mean = bert_features.mean(axis=0)
#   - 中心化: X_centered = bert_features - mean
#   - 计算协方差矩阵
#   - 特征值分解
#   - 选择前 128 个主成分

# 2. 推理阶段
new_texts = ["new movie description"]
new_bert_features = text_encoder.extract_features(new_texts)  # (1, 768)

# PCA 降维
embeddings = text_encoder.pca.transform(new_bert_features)  # (1, 128)
# 内部步骤：
#   - 中心化: new_bert_features - mean
#   - 投影: (new_bert_features - mean) @ components.T
```

### 图像编码中的 PCA

```python
# 1. 训练阶段
image_encoder = ImageEncoder(output_dim=128)
images = ["path/to/image1.png", "path/to/image2.png", ...]

# 提取 ResNet18 特征
resnet_features = image_encoder.extract_features(images)  # (N, 512)

# 训练 PCA
image_encoder.pca.fit(resnet_features)
# 将 512 维降到 128 维

# 2. 推理阶段
new_images = ["path/to/new_image.png"]
new_resnet_features = image_encoder.extract_features(new_images)  # (1, 512)

# PCA 降维
embeddings = image_encoder.pca.transform(new_resnet_features)  # (1, 128)
```

---

## 关键点总结

### 1. 训练一次，多次使用

- `fit()` 只在训练数据上调用一次
- `transform()` 可以在任意新数据上调用

### 2. 必须使用相同的均值和主成分

- 新数据必须用训练时的均值中心化
- 否则降维结果会不一致

### 3. 维度选择

- 本项目选择 128 维
- 可以根据需求调整（更小 = 更快但信息损失更多）

### 4. 信息保留

- PCA 会尽可能保留原始数据的方差
- 前 128 个主成分通常能保留 90%+ 的信息

### 5. sklearn 实现细节

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=128)
pca.fit(X_train)  # 训练
X_reduced = pca.transform(X_test)  # 降维

# 查看保留的方差比例
print(pca.explained_variance_ratio_.sum())  # 例如: 0.95 表示保留了 95% 的方差
```

---

## 数学公式总结

1. **中心化**: $\tilde{X} = X - \mu$
2. **协方差矩阵**: $C = \frac{1}{N-1}\tilde{X}^T\tilde{X}$
3. **特征值分解**: $C = V\Lambda V^T$
4. **降维**: $X_{reduced} = \tilde{X}W$，其中 $W$ 是前 k 个特征向量

---

## 可视化理解

### PCA 降维过程

```
原始数据 (768维)
    ↓
中心化 (减去均值)
    ↓
计算协方差矩阵 (768×768)
    ↓
特征值分解
    ↓
选择前128个主成分 (方向)
    ↓
投影到新空间
    ↓
降维后数据 (128维)
```

### 信息保留示意

```
原始维度: [1, 2, 3, ..., 768]
           ↓ PCA 选择最重要的维度
保留维度: [1, 2, 3, ..., 128]  (保留 ~95% 的信息)
丢弃维度: [129, 130, ..., 768] (只丢失 ~5% 的信息)
```

---

## 常见问题

### Q1: 为什么要降维？

**A**:

- 减少存储空间（768维 → 128维，节省 83% 空间）
- 加快计算速度（后续模型训练更快）
- 去除噪声（保留主要信息，过滤次要信息）

### Q2: 会损失多少信息？

**A**:

- 通常前 128 个主成分能保留 90-95% 的方差
- 可以通过 `pca.explained_variance_ratio_` 查看具体比例

### Q3: 如何选择降维后的维度？

**A**:

- 可以画 scree plot（碎石图）查看特征值分布
- 设置目标方差保留率（如 95%）
- 根据下游任务的性能调整

### Q4: PCA 和 BERT/ResNet 哪个更重要？

**A**:

- BERT/ResNet 提取语义特征（最重要）
- PCA 只是压缩，不改变语义
- 如果计算资源充足，可以不用 PCA，直接使用原始维度

---

## 参考资料

- [sklearn PCA 文档](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [PCA 原理详解](https://en.wikipedia.org/wiki/Principal_component_analysis)

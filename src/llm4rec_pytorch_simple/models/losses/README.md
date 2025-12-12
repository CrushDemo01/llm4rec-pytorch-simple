# `losses.py` 中的损失函数

## BCELoss（带 logits 的二元交叉熵）
- 适用场景：序列推荐中的点式学习（point-wise），对每个时间步的正样本与若干随机负样本做二分类。
- 计算方式：
  1. 对齐序列输出 `output_embeddings` 与正样本 `pos_embeddings`，点积得到正样本分数。
  2. 采样 `num_to_sample` 个负样本，点积得到负样本分数，并对负样本维度取平均。
  3. 用 `BCEWithLogitsLoss` 计算正样本（标签1）和负样本（标签0）损失，`supervision_mask` 屏蔽 padding，按有效位置求平均。
- 公式：
  $$
  \begin{aligned}
  s^+ &= \langle o, p \rangle, \\
  s^-_k &= \langle o, n_k \rangle, \quad k=1..K, \\
  \mathcal{L} &= m \cdot \Big[\;\text{BCE}(s^+, 1) + \frac{1}{K} \sum_{k=1}^K \text{BCE}(s^-_k, 0)\;\Big], \\
  	ext{loss} &= \dfrac{\sum \mathcal{L}}{\sum m}
  \end{aligned}
  $$
- 符号说明：$o$ 当前输出；$p$ 正样本；$n_k$ 第 $k$ 个负样本；$K=\text{num\_to\_sample}$；$\text{BCE}$ 为 `BCEWithLogitsLoss`；$m$ 为监督掩码；$\sum m$ 有效位置数。
- 配置要点：`num_to_sample` 控制负样本数；设大更丰富但占显存；采样避开正样本 ID。
- 优点：实现简单、收敛稳定、适合大规模负采样。
- 局限：点式目标未直接优化排序；随机负样本可能过易，需要更大采样量或更优采样策略。

## SampledSoftmaxLoss（采样 Softmax）
- 适用场景：序列推荐的近似多类/排序目标，把“1 正 + K 负”拼成局部 softmax。
- 计算方式：
  1. 点积得到正样本分数与 K 个负样本分数，拼接为局部 logits。
  2. 交叉熵以索引 0 为正样本类别，其余为负样本。
  3. 用 `supervision_mask` 屏蔽 padding，按有效位置求平均。
- 公式：
  $$
  \begin{aligned}
  s^+ &= \langle o, p \rangle, \\
  s^-_k &= \langle o, n_k \rangle, \quad k=1..K, \\
  	ext{logits} &= [s^+, s^-_1, \dots, s^-_K], \\
  y &= 0, \\
  \mathcal{L} &= m \cdot \text{CE}(\text{logits}, y), \\
  	ext{loss} &= \dfrac{\sum \mathcal{L}}{\sum m}
  \end{aligned}
  $$
- 符号说明：同上，$\text{CE}$ 为交叉熵；$y=0$ 表示正样本类别索引；$m$ 为监督掩码。
- 配置要点：`num_to_sample` 控制局部 softmax 宽度，越大近似全量 softmax 越好但开销增加；采样避开正样本 ID。
- 优点：较 BCE 更接近排序/多类信号，计算量与 $K$ 成正比，可调节效果与效率。
- 局限：仍是采样近似，难负样本质量依赖采样策略；大 $K$ 增加显存和算力需求。

## L2RegularizationLoss
- 适用场景：为模型参数添加 L2 正则（类似 weight decay），抑制过拟合。
- 计算方式：对所有参数平方求和乘以 `weight_decay` 系数，训练时与主损失相加。
- 公式：
  $$
  \mathcal{L}_{L2} = \lambda \sum_{\theta \in \mathcal{P}} \lVert \theta \rVert_2^2
  $$
- 符号说明：$\lambda$ 为 `weight_decay`；$\theta$ 为参数；$\mathcal{P}$ 为参数集合。
- 优点：实现简单、通用有效。
- 局限：无法解决难负样本或排序优化等问题，需要与主任务损失一起使用。

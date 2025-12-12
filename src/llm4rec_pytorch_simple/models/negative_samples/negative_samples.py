import torch


class LocalNegativeSamples(torch.nn.Module):
    def __init__(self, num_items: int):
        super().__init__()
        self.name = "LocalNegativeSamples"
        self.num_items = num_items
        self._item_emb: torch.nn.Embedding = None

    def forward(self, positive_ids: torch.Tensor, num_to_sample: int) -> torch.Tensor:
        """
        前向传播函数，计算负样本的 embeddings。

        参数:
        - positive_ids: 正样本的 ids，形状为 (batch_size,)。
        - num_to_sample: 要采样的负样本数量。

        返回:
        - neg_embeddings: 计算得到的负样本 embeddings，形状为 (batch_size, num_to_sample, embedding_dim)。
        """
        # 确定输出形状：与正样本形状相同，但最后一个维度是 num_to_sample
        output_shape = positive_ids.size() + (num_to_sample,)
        # torch.randint(low, high, size) 范围是 [1, num_items + 1)，因为 0 通常是 padding
        neg_ids = torch.randint(
            1, self.num_items + 1, size=output_shape, dtype=positive_ids.dtype, device=positive_ids.device
        )

        # 3. 防撞车逻辑 (Collision Check)
        # 为了比较 neg_ids 和 target_ids，需要对齐维度
        # positive: [..., L] -> [..., L, 1] -> 广播成 [..., L, K]
        positive_expanded = positive_ids.unsqueeze(-1).expand_as(neg_ids)

        mask = neg_ids == positive_expanded  # 找到撞车的位置
        if mask.any():
            # 对撞车的位置重新随机一个 (简单的 hack：直接 +1)
            # 这里的 % 运算是为了防止 +1 后超出 num_items 范围
            fixed = (neg_ids[mask] + 1) % self.num_items
            # 避免出现 0 (padding)
            fixed[fixed == 0] = 1
            neg_ids[mask] = fixed

        return neg_ids, self._item_emb(neg_ids)


if __name__ == "__main__":
    neg_samples = LocalNegativeSamples(num_items=1000)
    positive_ids = torch.randint(1, 1000, size=(2,))
    neg_ids = neg_samples(positive_ids, num_to_sample=5)
    print(positive_ids, neg_ids)

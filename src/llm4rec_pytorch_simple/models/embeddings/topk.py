import torch


class MIPSBruteForceTopK(torch.nn.Module):
    """Maximum Inner Product Search BruteForce - 通过暴力计算所有内积，找到最大的K个候选物品"""

    def forward(
        self,
        query_embeddings: torch.Tensor,
        item_embeddings_t: torch.Tensor,
        item_ids: torch.Tensor,
        k: int,
        sorted: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings: (B, ...). Implementation-specific.
            k: int. final top-k to return.
            sorted: bool. whether to sort final top-k results or not.

        Returns:
            Tuple of (top_k_scores x float, top_k_ids x int), both of shape (B, K,)
        """
        # (B, X,)
        all_logits = torch.mm(query_embeddings, item_embeddings_t)
        top_k_logits, top_k_indices = torch.topk(
            all_logits,
            dim=1,
            k=k,
            sorted=sorted,
            largest=True,
        )  # (B, k,)
        return top_k_logits, item_ids.squeeze(0)[top_k_indices]  # 将Top-K的索引映射回实际的物品ID

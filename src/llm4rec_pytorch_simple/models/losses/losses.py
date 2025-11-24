import torch
import torch.nn as nn

from llm4rec_pytorch_simple.utils.pylogger import RankedLogger

logger = RankedLogger(__name__)


class BCELoss(torch.nn.BCEWithLogitsLoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        logger.info(f"BCELoss initialized with reduction={reduction}")
        self.reduction = reduction
        self.name = "BCELoss"
        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(
        self, output_embeddings: torch.Tensor, pos_embeddings: torch.Tensor, neg_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播函数，计算 BCE 损失。

        参数:
        - output_embeddings: 模型输出的 embeddings，形状为 (batch_size, embedding_dim)。
        - pos_embeddings: 正样本的 embeddings，形状为 (batch_size, embedding_dim)。
        - neg_embeddings: 负样本的 embeddings，形状为 (batch_size, embedding_dim)。

        返回:
        - loss: 计算得到的 BCE 损失值。
        """
        pos_scores = (output_embeddings * pos_embeddings).sum(dim=-1)
        neg_scores = (output_embeddings * neg_embeddings).sum(dim=-1)
        loss = self.criterion(pos_scores, torch.ones_like(pos_scores)) + self.criterion(
            neg_scores, torch.zeros_like(neg_scores)
        )
        return loss


if __name__ == "__main__":
    loss = BCELoss()
    output_embeddings = torch.randn(2, 128)
    pos_embeddings = torch.randn(2, 128)
    neg_embeddings = torch.randn(2, 128)
    v = loss(output_embeddings, pos_embeddings, neg_embeddings)
    print(v)

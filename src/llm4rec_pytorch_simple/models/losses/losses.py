import torch
import torch.nn as nn
import torch.nn.functional as F

from llm4rec_pytorch_simple.models.negative_samples.negative_samples import LocalNegativeSamples
from llm4rec_pytorch_simple.utils.pylogger import RankedLogger

logger = RankedLogger(__name__)


class BCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "BCELoss"
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        negatives_sampler: LocalNegativeSamples,
        output_embeddings: torch.Tensor,
        pos_embeddings: torch.Tensor,
        positive_ids: torch.Tensor,  # 正样本id，形状为 (batch_size,)，用于采样负样本
        supervision_mask: torch.Tensor,  # mask，用于过滤掉填充位置
        num_to_sample: int,  # 每个正样本采样的负样本数量
    ) -> torch.Tensor:
        """
        前向传播函数，计算 BCE 损失。
        
        参数:
        - output_embeddings: 模型输出的 embeddings，形状为 (batch_size, embedding_dim)。
        - pos_embeddings: 正样本的 embeddings，形状为 (batch_size, embedding_dim)。
        - neg_embeddings: 负样本的 embeddings，形状为 (batch_size, embedding_dim)。
        - supervision_mask: 监督掩码，形状为 (batch_size, embedding_dim)。
        - num_to_sample: 每个正样本采样的负样本数量
        返回:
        - loss: 计算得到的 BCE 损失值。
        """
        # 步骤1：批量负采样
        # 为每个正样本采样多个负样本，形成对比学习的负例集合
        # num_to_sample通常为几百到几千，根据计算资源调整
        _sampled_ids, sampled_negative_embeddings = negatives_sampler(
            positive_ids=positive_ids,
            num_to_sample=num_to_sample,
        )

        # l2归一化
        output_embeddings = F.normalize(output_embeddings, dim=-1, p=2)
        pos_embeddings = F.normalize(pos_embeddings, dim=-1, p=2)
        sampled_negative_embeddings = F.normalize(sampled_negative_embeddings, dim=-1, p=2)  # shape [32, 200, 4, 64]

        pos_scores = (output_embeddings * pos_embeddings).sum(dim=-1)
        # 扩展 output_embeddings 以匹配 sampled_negative_embeddings 的维度
        # output_embeddings: [..., D] -> [..., 1, D]
        # sampled_negative_embeddings: [..., num_to_sample, D]
        # 广播后: [..., num_to_sample, D]
        output_embeddings_expanded = output_embeddings.unsqueeze(-2)  # [..., 1, D]
        neg_scores = (output_embeddings_expanded * sampled_negative_embeddings).sum(dim=-1)  # [..., num_to_sample]
        weighted_losses = (
            self.criterion(pos_scores, torch.ones_like(pos_scores))
            + self.criterion(neg_scores, torch.zeros_like(neg_scores)).mean(dim=-1)  # 对负样本维度求平均
        ) * supervision_mask
        loss = weighted_losses.sum() / supervision_mask.sum()
        return loss

# 网络权重正则化 loss-l2 范数
class L2RegularizationLoss(torch.nn.Module):
    def __init__(self, model, weight_decay):
        super().__init__()
        self.model = model
        self.weight_decay = weight_decay

    def forward(self):
        l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
        return self.weight_decay * l2_norm

if __name__ == "__main__":
    loss = BCELoss()
    output_embeddings = torch.randn(2, 128)
    pos_embeddings = torch.randn(2, 128)
    neg_embeddings = torch.randn(2, 128)
    v = loss(output_embeddings, pos_embeddings, neg_embeddings)
    print(v)

from typing import NamedTuple, Optional
from torch import Tensor


class TigerBatchDataStructure(NamedTuple):
    user_ids: Tensor
    item_ids: Tensor
    target_item_ids: Tensor
    seq_mask: Tensor


class TokenizedSeqBatch(NamedTuple):
    """Tokenized 序列批次，用于 TIGER 训练
    
    Attributes:
        user_ids: [B] 用户 ID
        sem_ids: [B, N*K] 历史序列的 semantic IDs（展平后）
        sem_ids_fut: [B, K] 目标物品的 semantic IDs（训练时）或 None（生成时）
        seq_mask: [B, N*K] 序列有效位置 mask，True = 有效
        token_type_ids: [B, N*K] 标记每个 token 在物品内的位置 (0,1,2,0,1,2,...)
        token_type_ids_fut: [B, K] 目标序列的 token type ids
    """
    user_ids: Tensor
    sem_ids: Tensor
    sem_ids_fut: Optional[Tensor]
    seq_mask: Tensor
    token_type_ids: Tensor
    token_type_ids_fut: Optional[Tensor]


class ModelOutput(NamedTuple):
    """模型输出数据结构
    
    Attributes:
        loss: 总体损失值（训练时使用）
        logits: 预测的 logits，形状为 [B, K+1, vocab_size]
        loss_d: [K] 每个位置的损失值（用于分析）
    """
    loss: Optional[Tensor]
    logits: Tensor
    loss_d: Optional[Tensor]


class GenerationOutput(NamedTuple):
    """生成模式输出数据结构
    
    Attributes:
        sem_ids: 生成的语义 ID 序列，形状为 [B, k, sem_id_dim]，k 为 beam size
        log_probas: 每个候选序列的对数概率，形状为 [B, k]
    """
    sem_ids: Tensor
    log_probas: Tensor


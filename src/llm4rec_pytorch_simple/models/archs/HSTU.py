import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from llm4rec_pytorch_simple.utils.pylogger import RankedLogger

# 注册 eval resolver 用于配置文件中的数学表达式
try:
    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    pass  # resolver 已经注册过了

logger = RankedLogger(__name__)


class HSTUBlock(nn.Module):
    """
    HSTU Block: Hierarchical Sequential Transduction Unit Block.
    参考官方实现: SequentialTransductionUnitJagged
    """
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        num_heads: int,
        activation_fn: str = "SiLU",
        dropout: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._linear_dim = linear_hidden_dim
        self._attention_dim = attention_dim
        self._num_heads = num_heads
        self._linear_activation = activation_fn.lower()
        self._dropout_ratio = dropout
        self._eps = eps

        # 使用 Parameter 而不是 Linear,参考官方实现
        self._uvqk = torch.nn.Parameter(
            torch.empty(
                embedding_dim,
                linear_hidden_dim * 2 * num_heads + attention_dim * num_heads * 2,
            ).normal_(mean=0, std=0.02)
        )

        self._o = torch.nn.Linear(
            in_features=linear_hidden_dim * num_heads, 
            out_features=embedding_dim,
        )
        torch.nn.init.xavier_uniform_(self._o.weight)

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        """对输入做 layer norm"""
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        """对 attention 输出做 layer norm"""
        return F.layer_norm(
            x, normalized_shape=[self._linear_dim * self._num_heads], eps=self._eps
        )

    def forward(self, x: torch.Tensor, rel_pos_bias: torch.Tensor = None):
        """
        Args:
            x: (B, L, D) 输入序列
            rel_pos_bias: 可选的相对位置偏置
        Returns:
            output: (B, L, D) 输出序列
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Norm -> MatMul -> Activation
        normed_x = self._norm_input(x)
        batched_mm_output = torch.mm(
            normed_x.view(-1, self._embedding_dim), 
            self._uvqk
        ).view(batch_size, seq_len, -1)
        
        if self._linear_activation == "silu":
            batched_mm_output = F.silu(batched_mm_output)
        elif self._linear_activation == "gelu":
            batched_mm_output = F.gelu(batched_mm_output)
        
        # 2. Split into u, v, q, k
        # 分成 4 部分: q(linear_hidden_dim * num_heads), k(attention_dim * num_heads), 
        #              v(attention_dim * num_heads), u(linear_hidden_dim * num_heads)
        u, v, q, k = torch.split(
            batched_mm_output,
            [
                self._linear_dim * self._num_heads,
                self._linear_dim * self._num_heads,
                self._attention_dim * self._num_heads,
                self._attention_dim * self._num_heads,
            ],
            dim=-1,
        )
        
        # 3. Reshape for multi-head attention
        # [B, L, H*D] -> [B, H, L, D]
        q = q.view(batch_size, seq_len, self._num_heads, self._attention_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self._num_heads, self._attention_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self._num_heads, self._linear_dim).transpose(1, 2)  # [B, H, L, linear_dim]
        
        # 4. Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self._attention_dim ** 0.5)
        
        if rel_pos_bias is not None:
            scores = scores + rel_pos_bias
        
        # 5. Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), 
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf')) # true是被mask的地方
        
        # 6. Softmax
        attn_weights = F.softmax(scores, dim=-1) # [B, H, L, L]
        
        # 7. Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [B, H, L, linear_dim]
        
        # 8. Reshape back: [B, H, L, linear_dim] -> [B, L, H*linear_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self._linear_dim * self._num_heads
        )
        
        # 9. u * norm(attn_output) - 这是官方实现的关键
        o_input = u * self._norm_attn_output(attn_output)
        
        # 10. Output projection with dropout and residual
        output = self._o(
            F.dropout(o_input, p=self._dropout_ratio, training=self.training)
        ) + x
        
        return output


class HSTUModel(nn.Module):
    """
    HSTU: Hierarchical Sequential Transduction Units model.
    """

    def __init__(
        self,
        max_sequence_len: int,  # 输入序列的最大长度
        embedding_dim: int,  # embedding的维度
        num_blocks: int,  # transformer的block数
        num_heads: int,  # transformer的head数
        attention_dim: int,  # transformer的attention维度
        linear_hidden_dim: int,  # transformer的linear hidden维度
        activation_fn: str = "SiLU",
        dropout: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        # 1. 模型中不用embedding，而是有embedding类处理，这样可以给不同模型都使用。

        self.max_sequence_len = max_sequence_len
        self.embedding_dim = embedding_dim
        self._eps = eps

        # 2. Transformer Encoder (核心骨干)
        self.transformer_encoder = nn.ModuleList([
            HSTUBlock(
                embedding_dim=embedding_dim,
                linear_hidden_dim=linear_hidden_dim,
                attention_dim=attention_dim,
                num_heads=num_heads,
                activation_fn=activation_fn,
                dropout=dropout,
                eps=eps,
            ) for _ in range(num_blocks)
        ])

        # 3. Final Norm
        self.final_norm = nn.LayerNorm(embedding_dim, eps=eps)

        # 初始化参数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化模型参数"""
        if isinstance(module, nn.Linear):
            # 使用 Xavier/Glorot 初始化
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, user_embeddings: torch.Tensor, past_ids: torch.Tensor = None):
        """
        Args:
            user_embeddings: (Batch, Seq_Len, Dim) 经过位置编码后的用户嵌入
            past_ids: (Batch, Seq_Len) 用户历史行为序列 ID (可选，用于 padding mask，暂未使用)
        Returns:
            output: (Batch, Seq_Len, Dim) Transformer 编码后的序列表示
        """
        # 通过所有 HSTU Block
        x = user_embeddings
        for block in self.transformer_encoder:
            x = block(x)  # 每个 block 内部处理因果掩码

        # 最终的 Layer Normalization
        output = self.final_norm(x)  # [B, N, D]

        return output


@hydra.main(version_base="1.1", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """
    HSTU 模型运行示例
    演示完整的前向传播流程：item_ids -> embeddings -> positional encoding -> HSTU -> output
    """
    logger.info("=" * 80)
    logger.info("HSTU Model Test")
    logger.info("=" * 80)

    # 1. 获取配置
    model_cfg = cfg.model
    data_cfg = cfg.data
    
    logger.info(f"\n模型配置:\n{model_cfg.sequence_model}")
    
    # 2. 实例化模型组件
    # 2.1 Item Embedding
    item_embedding = hydra.utils.instantiate(model_cfg.embedding)
    logger.info(f"\n✓ Item Embedding: {item_embedding}")
    
    # 2.2 位置编码
    positional_emb = hydra.utils.instantiate(model_cfg.positional_emb)
    logger.info(f"✓ Positional Embedding: {positional_emb}")
    
    # 2.3 HSTU 序列模型
    hstu_model: HSTUModel = hydra.utils.instantiate(model_cfg.sequence_model)
    logger.info(f"✓ HSTU Model: {hstu_model}")
    
    # 3. 准备测试数据
    BATCH_SIZE = 4
    SEQ_LEN = model_cfg.sequence_model.max_sequence_len
    NUM_ITEMS = data_cfg.num_items
    
    logger.info(f"\n测试数据配置:")
    logger.info(f"  Batch Size: {BATCH_SIZE}")
    logger.info(f"  Sequence Length: {SEQ_LEN}")
    logger.info(f"  Num Items: {NUM_ITEMS}")
    logger.info(f"  Embedding Dim: {model_cfg.item_embedding_dim}")
    
    # 3.1 生成模拟的 item IDs (包含 padding)
    past_ids = torch.randint(1, NUM_ITEMS, (BATCH_SIZE, SEQ_LEN))
    # 模拟不同长度的序列 (padding_idx=0)
    past_ids[0, 45:] = 0  # 第1个样本后5个位置padding
    past_ids[1, 40:] = 0  # 第2个样本后10个位置padding
    past_ids[2, 35:] = 0  # 第3个样本后15个位置padding
    # past_ids[3] 保持满序列
    
    logger.info(f"\n✓ 生成 past_ids: {past_ids.shape}")
    logger.info(f"  样本0有效长度: {(past_ids[0] != 0).sum().item()}")
    logger.info(f"  样本1有效长度: {(past_ids[1] != 0).sum().item()}")
    logger.info(f"  样本2有效长度: {(past_ids[2] != 0).sum().item()}")
    logger.info(f"  样本3有效长度: {(past_ids[3] != 0).sum().item()}")
    
    # 4. 完整的前向传播流程
    logger.info("\n" + "=" * 80)
    logger.info("开始前向传播")
    logger.info("=" * 80)
    
    # 4.1 Item Embedding
    item_embs = item_embedding(past_ids)  # [B, L, D]
    logger.info(f"\n步骤1 - Item Embedding:")
    logger.info(f"  Input shape: {past_ids.shape}")
    logger.info(f"  Output shape: {item_embs.shape}")
    
    # 4.2 添加位置编码
    user_embeddings, valid_mask = positional_emb(past_ids, item_embs)  # [B, L, D], [B, L, 1]
    logger.info(f"\n步骤2 - Positional Encoding:")
    logger.info(f"  Input shape: {item_embs.shape}")
    logger.info(f"  Output shape: {user_embeddings.shape}")
    logger.info(f"  Valid mask shape: {valid_mask.shape}")
    
    # 4.3 HSTU 模型前向传播
    hstu_output = hstu_model(user_embeddings, past_ids)  # [B, L, D]
    logger.info(f"\n步骤3 - HSTU Forward:")
    logger.info(f"  Input shape: {user_embeddings.shape}")
    logger.info(f"  Output shape: {hstu_output.shape}")
    
    # 5. 验证输出
    logger.info("\n" + "=" * 80)
    logger.info("输出验证")
    logger.info("=" * 80)
    
    # 5.1 检查输出形状
    assert hstu_output.shape == (BATCH_SIZE, SEQ_LEN, model_cfg.item_embedding_dim), \
        f"输出形状错误: 期望 {(BATCH_SIZE, SEQ_LEN, model_cfg.item_embedding_dim)}, 实际 {hstu_output.shape}"
    logger.info(f"✓ 输出形状正确: {hstu_output.shape}")
    
    # 5.2 检查输出值范围
    logger.info(f"\n输出统计:")
    logger.info(f"  Mean: {hstu_output.mean().item():.6f}")
    logger.info(f"  Std: {hstu_output.std().item():.6f}")
    logger.info(f"  Min: {hstu_output.min().item():.6f}")
    logger.info(f"  Max: {hstu_output.max().item():.6f}")
    
    # 5.3 提取最后一个时间步的输出 (用于预测下一个item)
    last_step_output = hstu_output[:, -1, :]  # [B, D]
    logger.info(f"\n✓ 最后时间步输出 (用于预测): {last_step_output.shape}")
    
    # 5.4 计算与所有 item embeddings 的相似度 (简化的推荐逻辑)
    all_item_embs = item_embedding.embeddings.weight[1:]  # [num_items, D], 排除 padding_idx=0
    scores = torch.matmul(last_step_output, all_item_embs.T)  # [B, num_items]
    top_k = 10
    top_k_items = torch.topk(scores, k=top_k, dim=-1).indices + 1  # [B, top_k], +1 因为排除了 padding
    
    logger.info(f"\n推荐结果 (Top-{top_k}):")
    for i in range(min(2, BATCH_SIZE)):  # 只显示前2个样本
        logger.info(f"  样本{i}: {top_k_items[i].tolist()}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ HSTU 模型测试完成!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

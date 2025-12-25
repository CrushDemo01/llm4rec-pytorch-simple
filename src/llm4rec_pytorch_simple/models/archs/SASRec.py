import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

from llm4rec_pytorch_simple.utils.pylogger import RankedLogger

logger = RankedLogger(__name__)


class SASRec(nn.Module):
    """
    Implements SASRec (Self-Attentive Sequential Recommendation, https://arxiv.org/abs/1808.09781, ICDM'18).
    """

    def __init__(
        self,
        max_sequence_len: int,  # 输入序列的最大长度
        embedding_dim: int,  # embedding的维度
        num_blocks: int,  # transformer的block数
        num_heads: int,  # transformer的head数
        ffn_hidden_extend: int,  # transformer的ffn hidden dim
        ffn_activation_fn: str = "gelu",  # transformer的ffn activation fn
        dropout: float = 0.1,
    ):
        super().__init__()
        # 1. 模型中不用embedding，而是有embedding类处理，这样可以给不同模型都使用。

        self.max_sequence_len = max_sequence_len
        self.embedding_dim = embedding_dim

        # 2. Transformer Encoder (核心骨干)
        # 使用 PyTorch 官方实现，包含 MultiheadAttention + FFN + LayerNorm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_extend * embedding_dim,
            activation=ffn_activation_fn,
            dropout=dropout,
            batch_first=True,  # 输入维度是 (Batch, Seq_Len, Dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)

        # 3. Final Norm
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # attention mask
        self.__init_att_mask()

        # 初始化参数 (Xavier)
        self.apply(self._init_weights)

    def __init_att_mask(self):
        """
        生成因果遮罩 (Upper Triangular Mask)。
        确保位置 i 只能看到 0...i 的信息，看不到 i+1 之后的信息。
        mask, 下三角矩阵为 1, 上半部分为 0:
        [[True,  False, False, False],   # 位置0只能看位置0
        [True,  True,  False, False],   # 位置1可以看位置0-1
        [True,  True,  True,  False],   # 位置2可以看位置0-2
        [True,  True,  True,  True]]    # 位置3可以看位置0-3
        """
        # 生成因果遮罩 (Upper Triangular Mask)
        # True 表示该位置被屏蔽 (mask out)，False 表示该位置可见
        # triu(diagonal=1) 生成上三角矩阵 (不含对角线)，即 j > i 的位置为 1 (True)
        mask = torch.triu(torch.ones(self.max_sequence_len, self.max_sequence_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, user_embeddings: torch.Tensor, past_ids: torch.Tensor):
        """
        Args:
            user_embeddings: (Batch, Seq_Len, Dim) 经过位置编码后的用户嵌入,就是序列 embedding
            past_ids: (Batch, Seq_Len) 用户历史行为序列 ID (用于生成 padding mask)
        Returns:
            output: (Batch, Seq_Len, Dim) Transformer 编码后的序列表示
        """
        _B, N, _D = user_embeddings.shape

        # 1. 生成因果掩码 (防止看到未来信息)
        # 从预先计算的 buffer 中切片取出当前序列长度的掩码
        causal_mask = self.causal_mask[:N, :N]  # 切片到实际序列长度 [N, N]

        # 2. 生成填充掩码 (屏蔽 padding 位置)
        # True 表示该位置是 padding，需要被屏蔽
        padding_mask = past_ids == 0  # [B, N]

        # 3. 通过 Transformer Encoder
        # src: 输入嵌入 [B, N, D]
        # mask: 因果掩码 [N, N]，控制位置间的注意力
        # src_key_padding_mask: 填充掩码 [B, N]，屏蔽 padding 位置
        output = self.transformer_encoder(
            src=user_embeddings,
            mask=causal_mask,  # 这个掩码会直接加到注意力分数上
            src_key_padding_mask=padding_mask,  # PyTorch 内部会用 masked_fill_ 将其转换
        )

        # 4. 最终的 Layer Normalization
        output = self.layer_norm(output)  # [B, N, D]

        return output


@hydra.main(version_base="1.1", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    model_cfg = cfg.model
    print(model_cfg)

    model: SASRec = hydra.utils.instantiate(model_cfg.sequence_model)

    # 模拟输入数据
    BATCH_SIZE = 2
    SEQ_LEN = 50
    EMBEDDING_DIM = model_cfg.sequence_model.embedding_dim

    # 模拟 past_ids (Batch, Seq_Len)
    past_ids = torch.randint(1, 100, (BATCH_SIZE, SEQ_LEN))
    past_ids[0, 40:] = 0  # 模拟第一个用户后面补零

    # 模拟 user_embeddings (Batch, Seq_Len, Dim)
    user_embeddings = torch.randn(BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM)

    # 前向传播
    output = model(user_embeddings, past_ids)

    print(f"Input IDs shape: {past_ids.shape}")  # [2, 50]
    print(f"Input Embeddings shape: {user_embeddings.shape}")  # [2, 50, 100]
    print(f"Output shape: {output.shape}")  # [2, 50, 100]

    # 简单验证：取最后一个时间步的输出
    last_step_output = output[:, -1, :]
    print(f"Last step output shape: {last_step_output.shape}")  # [2, 100]


if __name__ == "__main__":
    main()

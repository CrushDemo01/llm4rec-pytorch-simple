import torch
import torch.nn as nn

from llm4rec_pytorch_simple.utils.pylogger import RankedLogger

logger = RankedLogger(__name__)


class SASRec(nn.Module):
    def __init__(self, num_items, max_len, hidden_size, num_blocks, num_heads, dropout=0.1):
        super().__init__()

        self.max_len = max_len
        self.num_items = num_items
        self.hidden_size = hidden_size

        # 1. Embeddings
        # item_emb: [num_items + 1, hidden_size], +1 是为了留给 padding (index=0)
        self.item_emb = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.emb_dropout = nn.Dropout(dropout)

        # 2. Transformer Encoder (核心骨干)
        # 使用 PyTorch 官方实现，包含 MultiheadAttention + FFN + LayerNorm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,  # 输入维度是 (Batch, Seq_Len, Dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)

        # 3. Final Norm
        self.layer_norm = nn.LayerNorm(hidden_size)

        # 初始化参数 (Xavier)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_causal_mask(self, seq_len, device):
        """
        生成因果遮罩 (Upper Triangular Mask)。
        确保位置 i 只能看到 0...i 的信息，看不到 i+1 之后的信息。
        mask,下三角矩阵为 1，上半部分为 0：
        [[True,  False, False, False],   # 位置0只能看位置0
        [True,  True,  False, False],   # 位置1可以看位置0-1
        [True,  True,  True,  False],   # 位置2可以看位置0-2
        [True,  True,  True,  True]]    # 位置3可以看位置0-3
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)) == 1
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
        return mask

    def forward(self, input_ids):
        """
        Args:
            input_ids: (Batch, Seq_Len) 用户历史行为序列 ID
        Returns:
            logits: (Batch, Seq_Len, Num_Items + 1) 预测分数
        """
        seq_len = input_ids.size(1)
        device = input_ids.device

        # 1. 生成 Embedding
        # Positions: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

        # Input Embedding = Item Embedding + Positional Embedding
        x = self.item_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # 2. 生成 Mask 并通过 Transformer
        # src_mask: 这里的 mask 用于屏蔽未来信息
        # src_key_padding_mask: 用于屏蔽 padding (index=0) 的部分，防止其影响注意力计算
        causal_mask = self.get_causal_mask(seq_len, device)
        padding_mask = input_ids == 0  # True 代表是 padding，需要被 mask

        # Transformer Forward
        # 注意：nn.TransformerEncoder 自动处理了 Attention, FFN, Residual, Norm
        # Attention Score = Q \cdot K^T + causal_mask + padding_mask
        output = self.transformer_encoder(src=x, mask=causal_mask, src_key_padding_mask=padding_mask)

        output = self.layer_norm(output)  # torch.Size([2, 5, 64]) --> torch.Size([2, 5, 64])
        return output
        # 3. Prediction (计算相似度)
        # 原始论文中，使用 Transformer 的输出去点乘所有 Item 的 Embedding
        # 维度变化: (B, L, H) @ (Num_Items, H)^T -> (B, L, Num_Items)
        logits = torch.matmul(
            output, self.item_emb.weight.transpose(0, 1)
        )  # torch.Size([batch, seq_len, num_items+1]) logits[i, t, k] = batch_i在时间步t对物品k的预测分数
        return logits


# ==========================================
# 测试代码 (Demo)
# ==========================================
if __name__ == "__main__":
    # 参数配置
    NUM_ITEMS = 100  # 商品总数
    MAX_LEN = 5  # 序列最大长度
    HIDDEN_SIZE = 64  # 维度
    BATCH_SIZE = 2

    # 实例化模型
    model = SASRec(
        num_items=NUM_ITEMS,
        max_len=MAX_LEN,
        hidden_size=HIDDEN_SIZE,
        num_blocks=2,  # 2层 Transformer
        num_heads=2,  # 2个头
        dropout=0.1,
    )

    # 模拟输入数据 (Batch=2, Len=50)
    # 0 代表 padding
    dummy_input = torch.randint(1, NUM_ITEMS, (BATCH_SIZE, MAX_LEN))
    dummy_input[0, 40:] = 0  # 模拟第一个用户后面补零

    # 前向传播
    logits = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")  # [2, 50]
    print(f"Output shape: {logits.shape}")  # [2, 50, 101] (包含 padding 0)

    # 简单验证：取最后一个时间步的预测
    last_step_logits = logits[:, -1, :]
    print("last_step_logits:", last_step_logits, last_step_logits.shape)  # [2, 64]

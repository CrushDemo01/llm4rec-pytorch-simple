import math

import lightning as L
import torch
import torch.nn as nn

from llm4rec_pytorch_simple.utils.pylogger import RankedLogger

logger = RankedLogger(__name__)


class PositionalEmbedding(nn.Module):
    pass


class SinusoidalPositionalEncoding(nn.Module):
    """
    'Attention Is All You Need' 中的固定正余弦位置编码
    """

    def __init__(self, max_sequence_len: int, embedding_dim: int, dropout_rate: float):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._emb_dropout = nn.Dropout(p=dropout_rate)

        # --- 关键区别开始 ---
        # 不再创建 nn.Embedding，而是创建一个固定的矩阵 (Buffer)
        # 这个矩阵在训练过程中不会更新
        pe = torch.zeros(max_sequence_len, embedding_dim)
        position = torch.arange(0, max_sequence_len, dtype=torch.float).unsqueeze(1)

        # 计算分母中的频率项
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        # 填充偶数维度用 sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 填充奇数维度用 cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加一个 batch 维度: [1, max_len, dim]
        pe = pe.unsqueeze(0)

        # register_buffer 告诉 PyTorch 这是一个不需要计算梯度的张量，
        # 但是需要随着模型状态一起保存和加载。
        self.register_buffer("pe", pe)

    def forward(self, past_ids: torch.Tensor, past_embeddings: torch.Tensor):
        _B, N, _D = past_embeddings.size()

        # 1. 缩放 (和原来一样)
        user_embeddings = past_embeddings * math.sqrt(self._embedding_dim)

        # 2. 添加固定位置编码 (关键区别)
        # 直接从预计算好的 buffer 中切片取出前 N 个位置
        # self.pe 的形状是 [1, max_len, D]，可以自动广播到 [B, N, D]
        user_embeddings = user_embeddings + self.pe[:, :N, :]

        # 3. Dropout (和原来一样)
        user_embeddings = self._emb_dropout(user_embeddings)

        # 生成有效掩码，标记非填充位置。padding 位置是 0，所以 past_ids != 0 的位置是有效的（1）
        # .unsqueeze(-1) 将形状从[B, N]扩展为[B, N, 1]，以便与嵌入向量相乘
        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]

        # 将有效掩码应用到嵌入上，填充位置的嵌入变为0
        user_embeddings *= valid_mask
        return user_embeddings, valid_mask


class LearnablePositionalEmbedding(nn.Module):
    """
    可学习的位置编码
    
    注意: 这里继承 nn.Module 而不是 LightningModule,因为:
    1. 这只是一个简单的组件模块,不需要定义训练流程
    2. LightningModule 是用于定义完整的训练逻辑(training_step, validation_step等)
    3. 使用 nn.Module 更轻量,避免不必要的开销
    """
    def __init__(
        self,
        max_sequence_len: int,
        embedding_dim: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.max_sequence_len = max_sequence_len
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.positional_embedding = torch.nn.Embedding(max_sequence_len, embedding_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.reset_parameters()

    def reset_parameters(self):
        self.positional_embedding.weight.data.normal_(mean=0.0, std=self.embedding_dim**-0.5)

    def forward(
        self,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
    ):
        """
        前向传播函数，处理输入特征并返回预处理结果

        参数:
            past_ids (torch.Tensor): ID序列max_sequence_len，形状为[B, N]
            past_embeddings (torch.Tensor): 项目嵌入向量序列，形状为[B, N, D]

        返回:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
                - user_embeddings: 处理后的用户嵌入向量序列，形状为[B, N, D]
                - valid_mask: 有效位置掩码，形状为[B, N, 1]
                - None: 本预处理器不返回辅助掩码
        """
        B, N = past_ids.shape
        """
        将原始项目嵌入乘以 √D（D 是嵌入维度）,作用：缩放嵌入向量，防止位置编码在加法时被淹没
        torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1):生成位置索引 [B, N]，每行都是 [0, 1, 2, ..., N-1]
        """
        user_embeddings = past_embeddings * (self.embedding_dim**0.5) + self.positional_embedding(
            torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)
        )
        user_embeddings = self.dropout(user_embeddings)

        # 生成有效掩码，标记非填充位置。padding 位置是 0，所以 past_ids != 0 的位置是有效的（1）
        # .unsqueeze(-1) 将形状从[B, N]扩展为[B, N, 1]，以便与嵌入向量相乘
        valid_mask = (past_ids != 0).unsqueeze(-1).float()  # [B, N, 1]

        # 将有效掩码应用到嵌入上，填充位置的嵌入变为0
        user_embeddings *= valid_mask
        return user_embeddings, valid_mask


def run_example():
    # 1. 设置模拟数据
    batch_size = 2
    seq_len = 5
    embedding_dim = 8
    max_seq_len = 20
    dropout_rate = 0.1

    print(f"配置信息: Batch={batch_size}, SeqLen={seq_len}, Dim={embedding_dim}")

    # 模拟输入 ID (B, N)
    # 假设 0 是 padding，1-100 是有效 item id
    past_ids = torch.randint(1, 100, (batch_size, seq_len))
    # 手动制造一些 padding (例如第二个样本的最后两个位置是 padding)
    past_ids[1, -2:] = 0

    # 模拟输入 Embedding (B, N, D)
    past_embeddings = torch.randn(batch_size, seq_len, embedding_dim)

    print("\n" + "=" * 50)
    print("测试 1: SinusoidalPositionalEncoding (固定位置编码)")
    print("=" * 50)

    # 初始化模型
    sinusoidal_model = SinusoidalPositionalEncoding(
        max_sequence_len=max_seq_len, embedding_dim=embedding_dim, dropout_rate=dropout_rate
    )

    # 前向传播 (注意参数顺序: past_ids, past_embeddings)
    output_fixed, mask_fixed = sinusoidal_model(past_ids, past_embeddings)

    print(f"输入 Embeddings shape: {past_embeddings.shape}")
    print(f"输出 Embeddings shape: {output_fixed.shape}")
    print(f"输出 Mask shape: {mask_fixed.shape}")

    # 检查 padding 位置是否被 mask 为 0
    print("\n检查 Padding Mask 效果 (样本 1 的最后两个位置应为 0):")
    print(f"输入 IDs: {past_ids[1]}")
    print(f"输出向量片段 (最后两行): \n{output_fixed[1, -2:]}")

    print("\n" + "=" * 50)
    print("测试 2: LearnablePositionalEmbedding (可学习位置编码)")
    print("=" * 50)

    # 初始化模型
    learnable_model = LearnablePositionalEmbedding(
        max_sequence_len=max_seq_len, embedding_dim=embedding_dim, dropout_rate=dropout_rate
    )

    # 前向传播 (注意参数顺序: past_ids, past_embeddings)
    output_learn, mask_learn = learnable_model(past_ids, past_embeddings)

    print(f"输入 Embeddings shape: {past_embeddings.shape}")
    print(f"输出 Embeddings shape: {output_learn.shape}")
    print(f"输出 Mask shape: {mask_learn.shape}")

    print("\n检查 Padding Mask 效果 (样本 1 的最后两个位置应为 0):")
    print(f"输出向量片段 (最后两行): \n{output_learn[1, -2:]}")


if __name__ == "__main__":
    run_example()

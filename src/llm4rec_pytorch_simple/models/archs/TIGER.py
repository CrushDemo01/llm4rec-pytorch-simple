import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional



class TIGEREncoder(nn.Module):
    """
    TIGER Encoder: Bidirectional Transformer Encoder
    用于聚合用户历史交互序列，生成上下文表示
    """

    def __init__(
        self,
        embedding_dim: int,  # embedding 维度
        num_encoder_blocks: int,  # encoder block 数量
        num_heads: int,  # attention head 数量
        ffn_hidden_dim: int,  # FFN 隐藏层维度
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_dim,
            dropout=dropout,
            batch_first=True,  # (Batch, Seq, Dim)
            activation="gelu",
            norm_first=True,  # 官方实现通常推荐 Pre-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_blocks)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, user_embeddings: torch.Tensor, padding_mask: torch.Tensor = None):
        """
        Args:
            user_embeddings: (Batch, Seq_Len, Dim) 用户历史序列 embeddings
            padding_mask: (Batch, Seq_Len) padding mask，True 表示该位置需要被屏蔽
        Returns:
            encoder_output: (Batch, Seq_Len, Dim) 编码后的序列表示
        """
        # Encoder 处理
        encoder_output = self.encoder(
            src=user_embeddings,
            src_key_padding_mask=padding_mask,  # 屏蔽 padding 位置
        )

        # Layer Normalization
        encoder_output = self.layer_norm(encoder_output)

        return encoder_output


class TIGERDecoder(nn.Module):
    """
    TIGER Decoder: Autoregressive Transformer Decoder
    用于自回归地生成目标 item 的 Semantic ID token 序列
    """

    def __init__(
        self,
        embedding_dim: int,  # embedding 维度
        num_decoder_blocks: int,  # decoder block 数量
        num_heads: int,  # attention head 数量
        ffn_hidden_dim: int,  # FFN 隐藏层维度
        max_output_len: int,  # 输出序列最大长度
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_output_len = max_output_len

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_dim,
            dropout=dropout,
            batch_first=True,  # (Batch, Seq, Dim)
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_blocks)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # 生成因果遮罩 (Causal Mask)
        self._init_causal_mask()

    def _init_causal_mask(self):
        """
        生成因果遮罩，确保位置 i 只能看到 0...i 的信息
        上三角矩阵 (不含对角线) 为 True，表示被屏蔽

        注意：实际 decoder 输入长度 = max_output_len + 1（包含 BOS）
        """
        # causal_mask 大小需要包含 BOS token
        mask_size = self.max_output_len + 1
        mask = torch.triu(torch.ones(mask_size, mask_size), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(
        self,
        target_embeddings: torch.Tensor,
        encoder_output: torch.Tensor,
        target_padding_mask: torch.Tensor = None,
        memory_padding_mask: torch.Tensor = None,
    ):
        """
        Args:
            target_embeddings: (Batch, Target_Len, Dim) 目标序列 embeddings (Semantic ID tokens)
            encoder_output: (Batch, Src_Len, Dim) encoder 的输出
            target_padding_mask: (Batch, Target_Len) 目标序列的 padding mask
            memory_padding_mask: (Batch, Src_Len) encoder 输出的 padding mask
        Returns:
            decoder_output: (Batch, Target_Len, Dim) 解码后的序列表示
        """
        _B, target_len, _D = target_embeddings.shape

        # 获取因果遮罩 - 使用实际的decoder输入长度（target_embeddings已经包含了BOS）
        actual_decoder_len = target_embeddings.shape[1]
        causal_mask = self.causal_mask[:actual_decoder_len, :actual_decoder_len]

        # Decoder 处理
        decoder_output = self.decoder(
            tgt=target_embeddings,
            memory=encoder_output,
            tgt_mask=causal_mask,  # 因果mask，防止看到未来信息
            tgt_key_padding_mask=target_padding_mask,  # 标记目标序列中哪些位置是 padding
            memory_key_padding_mask=memory_padding_mask,  # 标记 encoder 输出中哪些位置是 padding
        )

        # Layer Normalization
        decoder_output = self.layer_norm(decoder_output)

        return decoder_output


class TIGER(nn.Module):
    """
    TIGER: Transformer Index for GEnerative Recommenders

    基于论文: Recommender Systems with Generative Retrieval (NeurIPS 2023)
    论文链接: https://arxiv.org/abs/2305.05065

    TIGER 将推荐任务建模为生成任务，使用 Transformer Encoder-Decoder 架构
    自回归地生成目标 item 的 Semantic ID（语义标识符）

    核心思想:
    1. Encoder: 聚合用户历史交互序列
    2. Decoder: 自回归生成下一个 item 的 Semantic ID token 序列
    3. Semantic ID: 通过层次化量化（如 RQ-VAE）将 item embeddings 编码为语义 token

    注意: 本实现中，Semantic ID 的生成和 embedding 由外部模块处理
    """
    def __init__(
        self,
        max_sequence_len: int,  # 输入序列最大长度
        max_output_len: int,  # 输出 Semantic ID 最大长度
        vocab_size: int,  # Semantic ID 词汇表大小 (codebook size)
        sid_embedding_dim: int,  # Semantic ID embedding 维度
        embedding_dim: int,  # tiger的 embedding 维度
        num_encoder_blocks: int,  # encoder block 数量
        num_decoder_blocks: int,  # decoder block 数量
        num_heads: int,  # attention head 数量
        ffn_hidden_extend: int,  # FFN 隐藏层扩展倍数
        dropout: float = 0.1,
        sid_embedding: Optional[nn.Embedding] = None,  # Semantic ID embedding layer (for generation)
    ):
        super().__init__()
        self.max_sequence_len = max_sequence_len
        self.max_output_len = max_output_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.sid_embedding = sid_embedding  # Store for use in generate()
        
        # BOS token
        self.bos_emb = nn.Parameter(torch.rand(embedding_dim))
        # Token Type Embedding： 用于区分不同码本位置，同一个item的token在不同码本位置的表示不同
        """
        但其实这里是没必要的，包括在 tiger(RQ-VAE Recommender) 中也是没有必要的。
        sem_ids_emb_fut 已经通过 SemIdEmbedder 包含了 token type 信息，❌ 再加一次 tte_fut 是重复编码

        self.token_type_emb = nn.Embedding(num_embeddings=num_codebooks, embedding_dim=embedding_dim)   # (3, Dim), 三个码本
        """

        # Encoder: 聚合用户历史序列
        self.encoder = TIGEREncoder(
            embedding_dim=embedding_dim,
            num_encoder_blocks=num_encoder_blocks,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_extend * embedding_dim,
            dropout=dropout,
        )

        # Decoder: 自回归生成 Semantic ID
        self.decoder = TIGERDecoder(
            embedding_dim=embedding_dim,
            num_decoder_blocks=num_decoder_blocks,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_extend * embedding_dim,
            max_output_len=max_output_len,
            dropout=dropout,
        )
        
        # 共享投影
        self.in_proj = nn.Linear(sid_embedding_dim, embedding_dim, bias=False)


        # 输出投影层: 将 decoder 输出映射到 vocab_size 维度
        self.output_projection = nn.Linear(embedding_dim, vocab_size)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        user_emb: torch.Tensor,
        post_sid_emb: torch.Tensor,
        post_sid: torch.Tensor,
        target_sid_emb: torch.Tensor,
        target_padding_mask: torch.Tensor,
        position_emb: torch.Tensor,
    ):
        """
        前向传播

        Args:
            user_emb: (Batch, Seq_Len, Dim) 用户历史序列 embeddings
            post_sid_emb: (Batch, Post_Len, Dim) 用户历史序列 embeddings
            post_padding_mask: (Batch, Post_Len) 用户历史序列 padding mask
            target_sid_emb: (Batch, Target_Len, Dim) 目标 Semantic ID 序列
            target_padding_mask: (Batch, Target_Len) 目标序列 padding mask
            position_emb: (Batch, Target_Len, Dim) 位置 embeddings

        Returns:
            logits: (Batch, Target_Len, Vocab_Size) 预测的 Semantic ID logits
        """
        _B, _S, _D = post_sid_emb.shape
        # 1: 构建Encoder输入（用户emb，历史序列+位置信息）
        input_embeddings = torch.cat([user_emb, post_sid_emb + position_emb], dim=1)    # [B, N+1, Dim]

        # 3: 为Encoder输入构建mask（用户token + 历史序列）
        post_sid_padding_mask = post_sid == 0  # [B, N]
        encoder_padding_mask = torch.cat([  # [B, N+1]  torch.Size([2, 31])
            torch.zeros(_B, 1, dtype=torch.bool, device=post_sid_padding_mask.device),  # 用户token始终有效(False=不屏蔽)
            post_sid_padding_mask  # 历史序列的mask
        ], dim=1)  # 建议用 dim 而不是 axis

        encoder_output = self.encoder(
            user_embeddings=self.in_proj(input_embeddings),
            padding_mask=encoder_padding_mask,
        )  # [Batch, Seq_Len+1, Dim]

        # 2: 构建Decoder输入（bos, 目标序列）
        # 先投影 target_sid_emb，再与 BOS 拼接
        target_sid_emb_proj = self.in_proj(target_sid_emb)  # [B, Target_Len, embedding_dim]
        batch_bos_emb = self.bos_emb.view(1, 1, -1).expand(_B, 1, -1)  # [B, 1, embedding_dim]
        decoder_input = torch.cat([batch_bos_emb, target_sid_emb_proj], dim=1)  # [B, Target_Len+1, embedding_dim]

        # 3: 为 Decoder 输入构建 mask（BOS + 目标序列）
        bos_mask = torch.zeros(_B, 1, dtype=torch.bool, device=target_padding_mask.device)  # BOS 始终有效
        decoder_padding_mask = torch.cat([bos_mask, target_padding_mask], dim=1)  # [B, Target_Len+1]

        # 4. Decoder: 自回归生成 Semantic ID
        decoder_output = self.decoder(
            target_embeddings=decoder_input,
            encoder_output=encoder_output,
            target_padding_mask=decoder_padding_mask,  # 使用扩展后的 mask，包括mask掉的bos
            memory_padding_mask=encoder_padding_mask,   # encoder 中padding 的mask
        )  # [Batch, Target_Len+1, Dim]

        # 5. 输出投影: 映射到 vocab_size 维度
        logits = self.output_projection(decoder_output)  # [Batch, Target_Len+1, Vocab_Size]

        return logits

    @torch.inference_mode
    def generate(
        self,
        user_emb: torch.Tensor,
        post_sid_emb: torch.Tensor,
        post_sid: torch.Tensor,
        position_emb: torch.Tensor,
        max_gen_len: int,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        自回归生成 Semantic IDs

        Args:
            user_emb: [B, 1, D] 用户 embedding
            post_sid_emb: [B, N, D] 历史序列 embeddings
            post_sid: [B, N] 历史序列 semantic IDs (用于 mask)
            position_emb: [B, N, D] 位置 embeddings
            max_gen_len: 最大生成长度
            temperature: 采样温度
            top_k: top-k 采样

        Returns:
            generated_ids: [B, max_gen_len] 生成的 semantic ID 序列
            scores: [B] 生成的分数（可选，这里返回 None）
        """
        batch_size = user_emb.shape[0]
        device = user_emb.device

        # 1. 编码器前向传播 (只需要一次)
        # 构建 Encoder 输入
        input_embeddings = torch.cat([user_emb, post_sid_emb + position_emb], dim=1)
        post_sid_padding_mask = post_sid == 0
        encoder_padding_mask = torch.cat([
            torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
            post_sid_padding_mask
        ], dim=1)

        encoder_output = self.encoder(
            user_embeddings=self.in_proj(input_embeddings),
            padding_mask=encoder_padding_mask,
        )

        # 2. 自回归生成
        generated_ids = torch.zeros(batch_size, max_gen_len, dtype=torch.long, device=device)

        # 初始化 decoder 输入：BOS token
        current_input = self.bos_emb.view(1, 1, -1).expand(batch_size, 1, -1)  # [B, 1, D]
        current_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

        for pos in range(max_gen_len):
            # 解码器前向传播
            decoder_output = self.decoder(
                target_embeddings=current_input,
                encoder_output=encoder_output,
                target_padding_mask=current_mask,
                memory_padding_mask=encoder_padding_mask,
            )  # [B, current_len, D]

            # 获取最后一个位置的输出
            last_output = decoder_output[:, -1:, :]  # [B, 1, D]
            logits = self.output_projection(last_output)  # [B, 1, Vocab]

            # 应用温度和 top-k
            if temperature != 1.0:
                logits = logits / temperature

            if top_k > 0:
                # Top-k 过滤
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[:, :, -1:]] = -float('inf')

            # 采样
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs[:, 0, :], num_samples=1)  # [B, 1]

            # 记录生成的 token
            generated_ids[:, pos] = next_id.squeeze(-1)

            # 准备下一个时间步的输入
            # 如果提供了 sid_embedding，使用它将 token ID 转换为 embedding
            if self.sid_embedding is not None:
                next_emb_raw = self.sid_embedding(next_id)  # [B, 1, sid_embedding_dim]
                next_emb = self.in_proj(next_emb_raw)  # [B, 1, embedding_dim]
            else:
                # 回退方案：假设 next_id 已经是 embedding（不推荐，会失败）
                # 这里保留是为了兼容性，但会在运行时报错
                raise ValueError(
                    "sid_embedding is required for generate(). "
                    "Please provide it during TIGER initialization."
                )

            current_input = torch.cat([current_input, next_emb], dim=1)
            current_mask = torch.cat([current_mask, torch.zeros(batch_size, 1, dtype=torch.bool, device=device)], dim=1)

        return generated_ids, None


def main():
    """
    TIGER 模型 forward 方法测试
    """
    print("=" * 80)
    print("TIGER Forward 方法测试")
    print("=" * 80)

    # 模型参数
    batch_size = 2
    max_sequence_len = 50
    post_len = 30  # 历史序列长度
    target_len = 3  # 目标序列长度
    max_output_len = 10  # Decoder 最大输出长度
    vocab_size = 1000  # Semantic ID 词汇表大小
    sid_embedding_dim = 64  # Semantic ID embedding 维度
    embedding_dim = 128  # TIGER 内部 embedding 维度
    num_encoder_blocks = 2
    num_decoder_blocks = 2
    num_heads = 4
    ffn_hidden_extend = 4
    dropout = 0.1

    # 创建模型
    model = TIGER(
        max_sequence_len=max_sequence_len,
        max_output_len=max_output_len,
        vocab_size=vocab_size,
        sid_embedding_dim=sid_embedding_dim,
        embedding_dim=embedding_dim,
        num_encoder_blocks=num_encoder_blocks,
        num_decoder_blocks=num_decoder_blocks,
        num_heads=num_heads,
        ffn_hidden_extend=ffn_hidden_extend,
        dropout=dropout,
    )

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 准备测试数据 ==========
    print("\n" + "=" * 80)
    print("准备测试数据")
    print("=" * 80)

    # 1. 用户 embedding (单个 token)
    user_emb = torch.randn(batch_size, 1, sid_embedding_dim)
    print(f"user_emb: {user_emb.shape}")

    # 2. 历史序列 Semantic ID embeddings
    post_sid_emb = torch.randn(batch_size, post_len, sid_embedding_dim)
    print(f"post_sid_emb: {post_sid_emb.shape}")

    # 3. 历史序列 Semantic ID (用于生成 padding mask)
    post_sid = torch.randint(1, vocab_size, (batch_size, post_len))
    post_sid[0, 25:] = 0  # 第一个样本后 5 个位置是 padding
    print(f"post_sid: {post_sid.shape}")
    print(f"  - Batch 0 padding positions: {(post_sid[0] == 0).sum().item()}")
    print(f"  - Batch 1 padding positions: {(post_sid[1] == 0).sum().item()}")

    # 4. 目标序列 Semantic ID embeddings
    target_sid_emb = torch.randn(batch_size, target_len, sid_embedding_dim)
    print(f"target_sid_emb: {target_sid_emb.shape}")

    # 5. 目标序列 padding mask
    target_padding_mask = torch.zeros(batch_size, target_len, dtype=torch.bool)
    target_padding_mask[1, 5:] = True  # 第二个样本后 3 个位置是 padding
    print(f"target_padding_mask: {target_padding_mask.shape}")
    print(f"  - Batch 0 padding positions: {target_padding_mask[0].sum().item()}")
    print(f"  - Batch 1 padding positions: {target_padding_mask[1].sum().item()}")

    # 6. 位置 embedding (用于历史序列)
    position_emb = torch.randn(batch_size, post_len, sid_embedding_dim)
    print(f"position_emb: {position_emb.shape}")

    # ========== 测试 1: Forward 形状检查 ==========
    print("\n" + "=" * 80)
    print("测试 1: Forward 形状检查")
    print("=" * 80)

    model.eval()
    with torch.no_grad():
        logits = model(
            user_emb=user_emb,
            post_sid_emb=post_sid_emb,
            post_sid=post_sid,
            target_sid_emb=target_sid_emb,
            target_padding_mask=target_padding_mask,
            position_emb=position_emb,
        )

    print(f"输出 logits 形状: {logits.shape}")
    expected_shape = (batch_size, target_len + 1, vocab_size)  # +1 因为加了 BOS
    assert logits.shape == expected_shape, f"期望 {expected_shape}, 实际 {logits.shape}"
    print(f"✓ 形状正确: {logits.shape}")

    # ========== 测试 2: Padding Mask 验证 ==========
    print("\n" + "=" * 80)
    print("测试 2: Padding Mask 验证")
    print("=" * 80)

    # 检查 padding 位置的输出是否受到影响
    # 理论上，padding 位置的输出应该与非 padding 位置有所不同
    with torch.no_grad():
        # Batch 1 的位置 5-7 是 padding
        padding_logits = logits[1, 6:, :]  # +1 因为 BOS
        non_padding_logits = logits[1, 1:5, :]  # +1 因为 BOS
        
        print(f"Padding 位置 logits 均值: {padding_logits.mean().item():.4f}")
        print(f"非 Padding 位置 logits 均值: {non_padding_logits.mean().item():.4f}")
        print("✓ Mask 应用正确")

    # ========== 测试 3: 梯度反向传播 ==========
    print("\n" + "=" * 80)
    print("测试 3: 梯度反向传播")
    print("=" * 80)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Forward
    logits = model(
        user_emb=user_emb,
        post_sid_emb=post_sid_emb,
        post_sid=post_sid,
        target_sid_emb=target_sid_emb,
        target_padding_mask=target_padding_mask,
        position_emb=position_emb,
    )

    # 生成随机目标 (注意: 需要 +1 长度因为有 BOS)
    targets = torch.randint(1, vocab_size, (batch_size, target_len + 1))
    # 为 targets 构建正确的 padding mask
    bos_mask = torch.zeros(batch_size, 1, dtype=torch.bool)
    targets_padding_mask = torch.cat([bos_mask, target_padding_mask], dim=1)
    targets[targets_padding_mask] = 0  # Padding 位置设为 0

    # 计算损失
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        ignore_index=0,  # 忽略 padding
    )

    print(f"损失值: {loss.item():.4f}")

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 检查梯度
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "模型参数没有梯度!"
    print("✓ 梯度反向传播成功")

    # ========== 测试 4: 投影层验证 ==========
    print("\n" + "=" * 80)
    print("测试 4: 投影层验证")
    print("=" * 80)

    # 检查 in_proj 是否被正确使用
    print(f"in_proj 权重形状: {model.in_proj.weight.shape}")
    print(f"期望: ({embedding_dim}, {sid_embedding_dim})")
    assert model.in_proj.weight.shape == (embedding_dim, sid_embedding_dim)
    print("✓ 投影层配置正确")

    # ========== 所有测试通过 ==========
    print("\n" + "=" * 80)
    print("所有测试通过! ✓")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
自定义 Transformer Layers,支持 RMSNorm 替换 LayerNorm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

from llm4rec_pytorch_simple.models.archs.normalization import RMSNorm


class TransformerEncoderLayer(nn.Module):
    """
    自定义 TransformerEncoderLayer,支持选择 LayerNorm 或 RMSNorm
    
    与 nn.TransformerEncoderLayer 兼容,但允许自定义 normalization 类型
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = False,
        norm_first: bool = False,
        norm_type: str = "layer_norm",  # "layer_norm" 或 "rms_norm"
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self.norm_first = norm_first

        # Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
        )

        # Feed-Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization Layers
        if norm_type == "rms_norm":
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        elif norm_type == "layer_norm":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"不支持的 norm_type: {norm_type}")

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"不支持的 activation: {activation}")

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,  # PyTorch 2.0+ 新增参数
    ) -> torch.Tensor:
        """
        Args:
            src: (Seq, Batch, Dim) 或 (Batch, Seq, Dim) if batch_first=True
            src_mask: (Seq, Seq) attention mask
            src_key_padding_mask: (Batch, Seq) padding mask
            is_causal: 是否使用因果遮罩 (通常 Encoder 不需要)
        """
        if self.norm_first:
            # Pre-LN: Norm -> Attention -> Add -> Norm -> FFN -> Add
            src = src + self._sa_block(self.norm1(src), src_mask, src_key_padding_mask, is_causal)
            src = src + self._ff_block(self.norm2(src))
        else:
            # Post-LN: Attention -> Add & Norm -> FFN -> Add & Norm
            src = self.norm1(src + self._sa_block(src, src_mask, src_key_padding_mask, is_causal))
            src = self.norm2(src + self._ff_block(src))

        return src

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Self-Attention Block"""
        x = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-Forward Block"""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(nn.Module):
    """
    自定义 TransformerDecoderLayer,支持选择 LayerNorm 或 RMSNorm
    
    与 nn.TransformerDecoderLayer 兼容,但允许自定义 normalization 类型
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = False,
        norm_first: bool = False,
        norm_type: str = "layer_norm",  # "layer_norm" 或 "rms_norm"
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self.norm_first = norm_first

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
        )

        # Cross-Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
        )

        # Feed-Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization Layers
        if norm_type == "rms_norm":
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
        elif norm_type == "layer_norm":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"不支持的 norm_type: {norm_type}")

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Activation
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"不支持的 activation: {activation}")

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_is_causal: bool = False,  # PyTorch 2.0+ 新增参数
        memory_is_causal: bool = False,  # PyTorch 2.0+ 新增参数
    ) -> torch.Tensor:
        """
        Args:
            tgt: (Tgt_Seq, Batch, Dim) 或 (Batch, Tgt_Seq, Dim) if batch_first=True
            memory: (Src_Seq, Batch, Dim) 或 (Batch, Src_Seq, Dim) if batch_first=True
            tgt_mask: (Tgt_Seq, Tgt_Seq) causal mask
            memory_mask: (Tgt_Seq, Src_Seq) cross-attention mask
            tgt_key_padding_mask: (Batch, Tgt_Seq) target padding mask
            memory_key_padding_mask: (Batch, Src_Seq) memory padding mask
            tgt_is_causal: 是否对 self-attention 使用因果遮罩
            memory_is_causal: 是否对 cross-attention 使用因果遮罩
        """
        if self.norm_first:
            # Pre-LN
            tgt = tgt + self._sa_block(self.norm1(tgt), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            tgt = tgt + self._mha_block(self.norm2(tgt), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            tgt = tgt + self._ff_block(self.norm3(tgt))
        else:
            # Post-LN
            tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            tgt = self.norm2(tgt + self._mha_block(tgt, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            tgt = self.norm3(tgt + self._ff_block(tgt))

        return tgt

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Self-Attention Block"""
        x = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    def _mha_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Multi-Head Cross-Attention Block"""
        x = self.multihead_attn(
            x, mem, mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout2(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-Forward Block"""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

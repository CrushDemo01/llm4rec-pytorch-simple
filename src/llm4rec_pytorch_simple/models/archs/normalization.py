import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    
    论文: Root Mean Square Layer Normalization (https://arxiv.org/abs/1910.07467)
    
    相比 LayerNorm:
    - 不减去均值,只做缩放归一化
    - 计算更简单,速度更快
    - 在 LLaMA、GPT-NeoX 等模型中广泛使用
    
    公式: y = (x / RMS(x)) * gamma
    其中: RMS(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        """
        Args:
            normalized_shape: 归一化的维度 (通常是 embedding_dim)
            eps: 数值稳定性的小常数
        """
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数 (对应 LayerNorm 的 weight)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., normalized_shape) 输入张量
        Returns:
            归一化后的张量,形状与输入相同
        """
        # 计算 RMS: sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # 归一化并缩放
        x_normed = x / rms
        
        return self.weight * x_normed

    def extra_repr(self) -> str:
        return f"normalized_shape={self.weight.shape[0]}, eps={self.eps}"

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class ResidualQuantizer(nn.Module):
    def __init__(self, num_codebooks: int, codebook_size: int, code_dim: int, distance_mode: str = 'l2'):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.distance_mode = distance_mode
        self.initialized = False  # 标记是否已初始化

        # 码本: (D, K, C)   torch.Size([2, 64, 32])
        # 初始化为随机值，后续会用k-means重新初始化
        self.codebooks = nn.Parameter(torch.randn(num_codebooks, codebook_size, code_dim))

    def initialize_codebooks_kmeans(self, z: torch.Tensor, max_iters: int = 100):
        """
        使用k-means聚类初始化码本，防止码本坍塌

        参数:
            z: 第一个batch的编码器输出 (batch_size, code_dim)
            max_iters: k-means最大迭代次数
        """
        if self.initialized:
            return

        with torch.no_grad():
            residual = z.clone()

            for i in range(self.num_codebooks):
                # 对当前残差进行k-means聚类
                centroids = self._kmeans(residual, self.codebook_size, max_iters)

                # 用聚类中心初始化码本
                self.codebooks.data[i] = centroids

                # 计算量化值并更新残差（为下一个码本准备）
                if self.distance_mode.lower() == 'l2':
                    dists = (
                        torch.sum(residual**2, dim=1, keepdim=True) +
                        torch.sum(centroids**2, dim=1) -
                        2 * torch.matmul(residual, centroids.t())
                    )
                elif self.distance_mode.lower() == 'cos':
                    residual_norm = F.normalize(residual, p=2, dim=1)
                    centroids_norm = F.normalize(centroids, p=2, dim=1)
                    dists = - torch.matmul(residual_norm, centroids_norm.t())
                else:
                    raise ValueError(f"Unsupported distance mode: {self.distance_mode}")

                indices = torch.argmin(dists, dim=1)
                z_q_i = centroids[indices]
                residual = residual - z_q_i

        self.initialized = True

    def _kmeans(self, x: torch.Tensor, k: int, max_iters: int = 100) -> torch.Tensor:
        """
        简单的k-means聚类实现

        参数:
            x: 输入数据 (n_samples, n_features)
            k: 聚类数量
            max_iters: 最大迭代次数

        返回:
            centroids: 聚类中心 (k, n_features)
        """
        n_samples = x.shape[0]

        # 如果样本数少于k，使用重复采样
        if n_samples < k:
            indices = torch.randint(0, n_samples, (k,), device=x.device)
            centroids = x[indices]
        else:
            # 随机选择k个样本作为初始中心
            indices = torch.randperm(n_samples, device=x.device)[:k]
            centroids = x[indices]

        for _ in range(max_iters):
            # 计算每个样本到各中心的距离
            if self.distance_mode.lower() == 'l2':
                dists = (
                    torch.sum(x**2, dim=1, keepdim=True) +
                    torch.sum(centroids**2, dim=1) -
                    2 * torch.matmul(x, centroids.t())
                )
            elif self.distance_mode.lower() == 'cos':
                x_norm = F.normalize(x, p=2, dim=1)
                centroids_norm = F.normalize(centroids, p=2, dim=1)
                dists = - torch.matmul(x_norm, centroids_norm.t())
            else:
                raise ValueError(f"Unsupported distance mode: {self.distance_mode}")

            # 分配样本到最近的中心
            labels = torch.argmin(dists, dim=1)

            # 更新中心
            new_centroids = torch.zeros_like(centroids)
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    new_centroids[j] = x[mask].mean(dim=0)
                else:
                    # 如果某个中心没有分配到样本，保持不变或随机重新初始化
                    new_centroids[j] = centroids[j]

            # 检查收敛
            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break

            centroids = new_centroids

        return centroids

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        参数:
            z: 输入张量 (batch_size, code_dim) 或 (batch_size, seq_len, code_dim)
        返回:
            z_q: 量化后的张量 (与 z 形状相同)
            codes: 索引 (batch_size, num_codebooks)
            commit_loss: 承诺损失
            codebook_loss: 码本损失
        """
        batch_size = z.shape[0]
        z_q = torch.zeros_like(z)   # torch.Size([10, 32])
        residual = z
        
        codes = []
        commit_loss = torch.tensor(0.0, device=z.device)
        codebook_loss = torch.tensor(0.0, device=z.device)
        
        for i in range(self.num_codebooks):
            # 寻找最近邻
            # residual: (B, C)
            # codebook[i]: (K, C)
            # Dist: (B, K)
            codebook = self.codebooks[i]    # torch.Size([64, 32])
            
            
            # 计算距离:
            if self.distance_mode.lower() == 'l2':
                # |x-y|^2 = |x|^2 + |y|^2 - 2xy
                dists = (
                    torch.sum(residual**2, dim=1, keepdim=True) +  # (10, 1)
                    torch.sum(codebook**2, dim=1) -                # (64,)
                    2 * torch.matmul(residual, codebook.t())       # (10, 64)
                )
            elif self.distance_mode.lower() == 'cos':
                # cos(x,y) = (x.y) / (|x||y|)
                # 归一化后计算余弦相似度，转换为距离
                residual_norm = F.normalize(residual, p=2, dim=1)      # (B, C)
                codebook_norm = F.normalize(codebook, p=2, dim=1)      # (K, C)
                dists = - torch.matmul(residual_norm, codebook_norm.t())  # (B, K)
            else:
                raise ValueError(f"Unsupported distance mode: {self.distance_mode}")
            
            # 获取索引
            indices = torch.argmin(dists, dim=1)  # torch.Size([10]) B
            codes.append(indices)
            
            # 获取量化值
            z_q_i = codebook[indices]  # (B, C) - 直接索引更简洁

            # 注意: 标准 VQ-VAE 损失有两部分:
            # 损失 1: 码本损失 ||sg[z] - e||²
                # - 让码本向量 e 靠近编码器输出 z
                # - sg[z] 意味着梯度不回传到编码器，只更新码本
            # 损失 2: 承诺损失 ||z - sg[e]||²
                # - 让编码器输出 z 靠近码本向量 e
                # - sg[e] 意味着梯度不回传到码本，只更新编码器
            
            # 码本损失：让码本向量靠近编码器输出（不回传到编码器）
            codebook_loss += F.mse_loss(z_q_i, residual.detach())
            
            # 承诺损失：让编码器输出靠近码本向量（不回传到码本）
            commit_loss += F.mse_loss(z_q_i.detach(), residual)
            
            # 更新残差
            residual = residual - z_q_i
            z_q = z_q + z_q_i   # 累加量化结果
        # codes是torch.Size([10])，两个，从dim=1堆叠，得到 torch.Size([10, 2])
        codes = torch.stack(codes, dim=1)  # (B, num_codebooks) torch.Size([10, 2])
        
        # 直通估计器
        """
        # 展开
        z_q_new = z + (z_q - z)  # detach 在前向传播中不影响值
                = z + z_q - z
                = z_q  # ✅ 前向传播使用量化后的值

        # 梯度计算
        ∂L/∂z = ∂L/∂z_q_new * ∂z_q_new/∂z
        # 因为 z_q_new = z + (z_q - z).detach()
        # (z_q - z).detach() 的梯度为 0
        ∂z_q_new/∂z = 1 + 0 = 1  # ✅ 梯度直接传递
        # 所以
        ∂L/∂z = ∂L/∂z_q_new  # 梯度完全相同，"直通"了！
        """
        z_q = z + (z_q - z).detach()    # 将 z_q - z 的梯度设为 0

        return z_q, codes, commit_loss, codebook_loss

class RQVAE(nn.Module):
    def __init__(
        self, 
        input_dim: int = 256,  # 从 896 更新为 256 (128+128)
        hidden_dim: List[int] = [512],
        code_dim: int = 128,  # 从 256 更新为 128
        num_codebooks: int = 3, 
        codebook_size: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        
        # 编码器
        dims = [self.input_dim] + self.hidden_dim + [self.code_dim]
        encoder_layers = []
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # 最后一层不加 normalization 和 dropout
                encoder_layers.append(nn.LayerNorm(dims[i+1]))
                encoder_layers.append(nn.GELU())
                encoder_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 量化器
        self.quantizer = ResidualQuantizer(num_codebooks, codebook_size, code_dim)
        
        # 解码器
        dims_reversed = dims[::-1]
        decoder_layers = []
        for i in range(len(dims_reversed) - 1):
            decoder_layers.append(nn.Linear(dims_reversed[i], dims_reversed[i+1]))
            if i < len(dims_reversed) - 2:  # 最后一层不加 normalization 和 dropout
                decoder_layers.append(nn.LayerNorm(dims_reversed[i+1]))
                decoder_layers.append(nn.GELU())
                decoder_layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*decoder_layers)

        # 初始化模型参数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化模型参数"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        参数:
            x: 输入特征 (batch_size, input_dim)
        返回:
            codes: 离散编码
            x_recon: 重构特征
            commit_loss: 承诺损失
            codebook_loss: 码本损失
        """
        z = self.encoder(x)
        z_q, codes, commit_loss, codebook_loss = self.quantizer(z)
        x_recon = self.decoder(z_q)
        return codes, x_recon, commit_loss, codebook_loss

if __name__ == "__main__":
    # 测试
    print("Testing RQVAE...")
    emd_dim = 32
    x = torch.randn(10, emd_dim)
    
    model = RQVAE(
        input_dim=32,           # 你的 PCA 降维维度
        hidden_dim=[32],
        code_dim=32,          # 与输入维度相同或稍大即可
        num_codebooks=2,        # 2 层足够：64^2 = 4096 > 3884
        codebook_size=64,       # 降低到 64：64^2 = 4096
    )
    
    codes, x_recon, commit_loss, codebook_loss = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Recon shape: {x_recon.shape}")
    print(f"Codes shape: {codes.shape}")
    print(f"Commit loss: {commit_loss.item()}")
    print(f"Codebook loss: {codebook_loss.item()}")
    
    assert x_recon.shape == x.shape
    assert codes.shape == (10, 2)

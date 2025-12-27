"""
多模态编码器模块 - 完全基于 nn.Module 实现

本模块实现了两个编码器：
1. TextEncoder: 使用 BERT + PCA 将文本转换为固定维度的向量
2. ImageEncoder: 使用 ResNet18 + PCA 将图像转换为固定维度的向量
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertModel
from PIL import Image
import numpy as np
from typing import List, Tuple
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf


class TorchPCA(nn.Module):
    """
    基于 PyTorch 的 PCA 实现，支持 GPU 加速
    注意：PCA 的 fit() 需要一次性看到所有数据，不支持 batch fit
    """
    def __init__(self, n_components: int):
        super().__init__()
        self.n_components = n_components

        # 使用 register_buffer 注册参数，这样 .to(device) 会自动移动它们
        self.register_buffer('mean_', None)
        self.register_buffer('components_', None)
        self.register_buffer('explained_variance_', None)

    def fit(self, X: torch.Tensor) -> 'TorchPCA':
        """
        训练 PCA - 必须一次性提供所有数据
        Args:
            X: torch.Tensor, shape (n_samples, n_features)
        Returns:
            self
        """
        n_samples, n_features = X.shape

        # 步骤 1: 计算均值并中心化
        mean = X.mean(dim=0)
        X_centered = X - mean

        # 步骤 2: SVD 分解
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

        # 步骤 3: 选择前 n_components 个主成分
        components = Vt[:self.n_components, :]

        # 步骤 4: 计算方差
        explained_variance = (S[:self.n_components] ** 2) / (n_samples - 1)

        # 注册为 buffer
        self.register_buffer('mean_', mean)
        self.register_buffer('components_', components)
        self.register_buffer('explained_variance_', explained_variance)

        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        降维 - 支持 batch 推理
        Args:
            X: shape (batch_size, n_features)
        Returns:
            shape (batch_size, n_components)
        """
        if self.mean_ is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        # 中心化并投影
        return (X - self.mean_) @ self.components_.T


class TextEncoder(nn.Module):
    """
    文本编码器: BERT + PCA
    支持 batch 推理
    """
    def __init__(self, output_dim: int = 128, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.output_dim = output_dim
        self.model_name = model_name

        # BERT tokenizer (不是 nn.Module，不需要注册)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # BERT model
        self.bert = BertModel.from_pretrained(model_name)
        self.bert.eval()  # 冻结 BERT，只用于特征提取

        # PCA
        self.pca = TorchPCA(n_components=output_dim)

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        前向传播 - 对一个 batch 的文本进行编码，不包含 PCA

        Args:
            texts: 文本列表，例如 ["movie 1", "movie 2", ...]
                   shape: (batch_size,)

        Returns:
            shape (batch_size, bert_hidden_size) - 通常是 768
        """
        # 1. 分词 + padding + 生成 attention_mask
        encoded = self.tokenizer(
            texts,
            padding=True,        # 自动 padding 到 batch 中最长的长度
            truncation=True,     # 截断超过 512 的文本
            max_length=512,      # BERT 最大长度
            return_tensors='pt'  # 返回 PyTorch tensor
        )

        # 将 input_ids 和 attention_mask 移到与 BERT 相同的设备
        device = next(self.bert.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # 2. BERT 提取特征
        with torch.no_grad():
            outputs = self.bert(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            )
            # 使用 [CLS] token 的嵌入（第一个 token）
            cls_features = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

        return cls_features


class ImageEncoder(nn.Module):
    """
    图像编码器: ResNet18 + PCA
    """
    def __init__(
        self,
        output_dim: int = 128,
        resize: int = 256,
        crop_size: int = 224,
        normalize_mean: List[float] = [0.485, 0.456, 0.406],
        normalize_std: List[float] = [0.229, 0.224, 0.225]
    ):
        super().__init__()
        self.output_dim = output_dim

        # ResNet18 (移除最后的分类层)
        # 使用 weights 参数替代 deprecated 的 pretrained 参数
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # 输出 512 维
        self.resnet.eval()  # 冻结 ResNet

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ])

        # PCA
        self.pca = TorchPCA(n_components=output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 支持 batch 推理
        Args:
            images: shape (batch_size, 3, 224, 224) - 已经预处理过的 tensor
        Returns:
            shape (batch_size, 512)
        """
        # ResNet 提取特征
        with torch.no_grad():
            features = self.resnet(images)  # (batch, 512, 1, 1)
            features = features.squeeze(-1).squeeze(-1)  # (batch, 512)

        return features


class ImageDataset(Dataset):
    """
    图像数据集，用于 DataLoader 并行加载
    """
    def __init__(self, image_files: List[Path], transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            # 从文件名提取 MovieID (例如 "1.png" -> 1)
            movie_id = int(img_path.stem)
            return img, movie_id
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 返回 None，需要在 collate_fn 中处理，或者这里返回一个占位符
            # 为简单起见，这里假设数据大部分是好的，返回一个全 0 张量和 -1 ID
            # 实际生产中应该用 collate_fn 过滤
            return torch.zeros((3, 224, 224)), -1


def get_device(device_config: str) -> torch.device:
    """
    根据配置获取设备

    Args:
        device_config: 设备配置字符串 ("cuda", "mps", "cpu")

    Returns:
        torch.device 对象
    """
    if device_config == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_config == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def extract_text_features(
    encoder: TextEncoder,
    df: pd.DataFrame,
    text_column: str,
    batch_size: int,
    device: torch.device
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    提取文本特征并进行 PCA 降维

    Args:
        encoder: 文本编码器
        df: 包含文本数据的 DataFrame
        text_column: 文本列名
        batch_size: 批次大小
        device: 计算设备

    Returns:
        (embeddings, movie_ids): 降维后的嵌入和对应的电影 ID
    """
    print(f"\n[提取文本特征] 使用 {encoder.model_name}")

    # 第一步：提取 BERT 特征
    all_features = []
    for i in tqdm(range(0, len(df), batch_size), desc="BERT 编码"):
        batch_texts = df[text_column].iloc[i:i+batch_size].tolist()
        # 移除可能的 NaN 值
        batch_texts = [str(t) if pd.notna(t) else "" for t in batch_texts]
        # 编码并移到 CPU 节省显存
        features = encoder(batch_texts).cpu()
        all_features.append(features)

    features = torch.cat(all_features, dim=0)
    print(f"  - BERT 特征形状: {features.shape}")

    # 第二步：PCA 降维
    print(f"  - PCA 降维: {features.shape[1]} -> {encoder.output_dim}")
    encoder.pca.fit(features.to(device))
    embeddings = encoder.pca(features.to(device)).cpu()

    # 打印前 5 个主成分的解释方差
    explained_var = encoder.pca.explained_variance_[:5].cpu().numpy()
    print(f"  - 解释方差 (前5个): {explained_var}")

    movie_ids = df['movieid'].values
    return embeddings, movie_ids


def extract_image_features(
    encoder: ImageEncoder,
    image_dir: Path,
    batch_size: int,
    device: torch.device
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    提取图像特征并进行 PCA 降维

    Args:
        encoder: 图像编码器
        image_dir: 图像目录路径
        batch_size: 批次大小
        device: 计算设备

    Returns:
        (embeddings, movie_ids): 降维后的嵌入和对应的电影 ID
    """
    print(f"\n[提取图像特征] 从目录: {image_dir}")

    image_files = sorted(image_dir.glob("*.png"))
    print(f"  - 找到 {len(image_files)} 张图像")

    # 第一步：提取 ResNet 特征
    all_features = []
    movie_ids = []

    # 使用 DataLoader 并行加载
    dataset = ImageDataset(image_files, encoder.transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,  # 使用多进程加载
        pin_memory=True if device.type == 'cuda' else False
    )

    for batch_images, batch_ids in tqdm(dataloader, desc="ResNet 编码"):
        # 过滤掉加载失败的数据 (id == -1)
        valid_mask = batch_ids != -1
        if not valid_mask.any():
            continue
            
        batch_images = batch_images[valid_mask].to(device)
        batch_ids = batch_ids[valid_mask]

        features = encoder(batch_images).cpu()
        all_features.append(features)
        movie_ids.extend(batch_ids.numpy())

    features = torch.cat(all_features, dim=0)
    print(f"  - ResNet 特征形状: {features.shape}")

    # 第二步：PCA 降维
    print(f"  - PCA 降维: {features.shape[1]} -> {encoder.output_dim}")
    encoder.pca.fit(features.to(device))
    embeddings = encoder.pca(features.to(device)).cpu()

    # 打印前 5 个主成分的解释方差
    explained_var = encoder.pca.explained_variance_[:5].cpu().numpy()
    print(f"  - 解释方差 (前5个): {explained_var}")

    return embeddings, np.array(movie_ids)


def save_embeddings(
    embeddings: torch.Tensor,
    movie_ids: np.ndarray,
    output_path: Path,
    output_dim: int
) -> None:
    """
    保存嵌入为 Parquet 格式

    Args:
        embeddings: 嵌入张量 (N, output_dim)
        movie_ids: 电影 ID 数组
        output_path: 输出文件路径
        output_dim: 嵌入维度
    """
    embeddings_np = embeddings.numpy()

    df = pd.DataFrame({
        'movie_id': movie_ids,
        **{f'emb_{i}': embeddings_np[:, i] for i in range(output_dim)}
    })

    df.to_parquet(output_path, index=False, compression='snappy')
    print(f"  - 已保存: {output_path}")


def main(cfg: DictConfig) -> None:
    """
    主函数，通过 Hydra 配置启动编码器训练

    Args:
        cfg: Hydra 配置对象，自动从配置文件加载
    """
    # ==================== 配置解析 ====================
    data_dir = Path(cfg.model.data_dir)
    text_csv = data_dir / cfg.model.text_csv
    image_dir = data_dir / cfg.model.image_dir
    output_dir = data_dir / cfg.model.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    output_dim = cfg.model.output_dim
    text_model_name = cfg.model.text_model_name
    text_batch_size = cfg.model.text_batch_size
    image_batch_size = cfg.model.image_batch_size

    # 设备配置
    device = get_device(cfg.model.device)

    # 打印配置信息
    print("=" * 60)
    print("多模态编码器训练")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"输出维度: {output_dim}")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"文本模型: {text_model_name}")
    print("=" * 60)

    # ==================== 文本编码 ====================
    print("\n[1/4] 加载文本数据...")
    df_text = pd.read_csv(text_csv)
    print(f"  - 加载了 {len(df_text)} 条文本记录")

    print("\n[2/4] 初始化文本编码器并提取特征...")
    text_encoder = TextEncoder(
        output_dim=output_dim,
        model_name=text_model_name
    ).to(device)

    text_embeddings, text_movie_ids = extract_text_features(
        encoder=text_encoder,
        df=df_text,
        text_column='text',
        batch_size=text_batch_size,
        device=device
    )

    # 保存文本嵌入
    text_output_path = output_dir / f"text_embeddings.parquet"
    save_embeddings(text_embeddings, text_movie_ids, text_output_path, output_dim)

    # ==================== 图像编码 ====================
    print("\n[3/4] 初始化图像编码器并提取特征...")
    image_encoder = ImageEncoder(
        output_dim=output_dim,
        resize=cfg.model.image.resize,
        crop_size=cfg.model.image.crop_size,
        normalize_mean=cfg.model.image.normalize_mean,
        normalize_std=cfg.model.image.normalize_std
    ).to(device)

    image_embeddings, image_movie_ids = extract_image_features(
        encoder=image_encoder,
        image_dir=image_dir,
        batch_size=image_batch_size,
        device=device
    )

    # 保存图像嵌入
    image_output_path = output_dir / f"image_embeddings.parquet"
    save_embeddings(image_embeddings, image_movie_ids, image_output_path, output_dim)

    # ==================== 保存模型 ====================
    print("\n[4/4] 保存编码器模型...")
    model_path = output_dir / "encoders.pt"
    torch.save({
        'text_encoder': text_encoder.state_dict(),
        'image_encoder': image_encoder.state_dict(),
        'config': OmegaConf.to_container(cfg, resolve=True)
    }, model_path)
    print(f"  - 已保存: {model_path}")

    # ==================== 总结 ====================
    print("\n" + "=" * 60)
    print("✅ 完成！")
    print("=" * 60)
    print(f"文本嵌入: {text_output_path}")
    print(f"图像嵌入: {image_output_path}")
    print(f"编码器模型: {model_path}")
    print("=" * 60)


@hydra.main(version_base="1.2", config_path="../configs/model", config_name="encoders.yaml")
def hydra_main(cfg: DictConfig) -> None:
    """
    Hydra 入口函数，自动加载配置文件

    Args:
        cfg: Hydra 配置对象，自动从配置文件加载
    """
    main(cfg)


if __name__ == "__main__":
    # 使用 Hydra 配置
    hydra_main()

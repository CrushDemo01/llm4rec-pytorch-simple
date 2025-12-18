"""
多模态编码器模块 - 完全基于 nn.Module 实现

本模块实现了两个编码器：
1. TextEncoder: 使用 BERT + PCA 将文本转换为固定维度的向量
2. ImageEncoder: 使用 ResNet18 + PCA 将图像转换为固定维度的向量
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertModel
from PIL import Image
import numpy as np
from typing import List
import pandas as pd
from pathlib import Path
from tqdm import tqdm

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
    
    def fit(self, X: torch.Tensor):
        """
        训练 PCA - 必须一次性提供所有数据
        Args:
            X: torch.Tensor, shape (n_samples, n_features)
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
        前向传播 - 对一个 batch 的文本进行编码，这里不做 pca
        
        Args:
            texts: 文本列表，例如 ["movie 1", "movie 2", ...]
                   shape: (batch_size,)
        
        Returns:
            shape (batch_size, output_dim)
        """
        # 1. 分词 + padding + 生成 attention_mask
        encoded = self.tokenizer(
            texts,
            padding=True,        # 自动 padding 到 batch 中最长的长度
            truncation=True,     # 截断超过 512 的文本
            max_length=512,      # BERT 最大长度
            return_tensors='pt'  # 返回 PyTorch tensor
        )
        # encoded 包含:
        # - input_ids: (batch_size, seq_len) - token IDs
        # - attention_mask: (batch_size, seq_len) - 1=真实token, 0=padding
        
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
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim
        
        # ResNet18 (移除最后的分类层)
        resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # 输出 512 维
        self.resnet.eval()  # 冻结 ResNet
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

def main():
    DATA_DIR = Path("/Users/yyds/workspace/REC/llm4rec-pytorch-simple/ml-1m/multimodal_datasets")
    TEXT_CSV = DATA_DIR / "text.csv"
    IMAGE_DIR = DATA_DIR / "image"
    OUTPUT_DIR = DATA_DIR / "embeddings_32d"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    OUTPUT_DIM = 32  # PCA 降维目标维度
    DEVICE = torch.device("cpu")
    batch_size = 16
    
    # ==================== 1. 加载文本数据 ====================
    df_text = pd.read_csv(TEXT_CSV)
    print(f"  - 加载了 {len(df_text)} 条文本记录")
    
    # ==================== 2. 初始化文本编码器 ====================
    text_encoder = TextEncoder(output_dim=OUTPUT_DIM).to(DEVICE)
    
    # ==================== 3. 提取文本特征 (BERT) ====================
    all_text_features = []
    
    for i in tqdm(range(0, len(df_text), batch_size), desc="文本编码"):
        batch_texts = df_text['description'].iloc[i:i+batch_size].tolist()
        # 移除可能的 NaN 值
        batch_texts = [str(t) if pd.notna(t) else "" for t in batch_texts]
        # 先转入 cpu ，节省内存
        # 编码 (BERT 输出 768 维)
        features = text_encoder(batch_texts).cpu()  # (batch, 768)
        all_text_features.append(features)
    
    text_features = torch.cat(all_text_features, dim=0)  # (N, 768)
    print(f"  - 文本特征形状: {text_features.shape}")
    
    # ==================== 4. PCA 降维文本特征 ====================
    text_encoder.pca.fit(text_features.to(DEVICE))
    text_embeddings = text_encoder.pca(text_features.to(DEVICE)).cpu()  # (N, 32)
    print(f"  - 降维后形状: {text_embeddings.shape}")
    print(f"  - 解释方差比: {text_encoder.pca.explained_variance_[:5].cpu().numpy()}")
    
    # 保存文本嵌入为 Parquet
    movie_ids = df_text['MovieID'].values
    text_emb_np = text_embeddings.numpy()
    
    df_text_emb = pd.DataFrame({
        'movie_id': movie_ids,
        **{f'emb_{i}': text_emb_np[:, i] for i in range(OUTPUT_DIM)}  # 展开为列
    })
    df_text_emb.to_parquet(
        OUTPUT_DIR / "text_embeddings_32d.parquet", 
        index=False,
        compression='snappy'
    )
    print(f"  - 已保存到: {OUTPUT_DIR / 'text_embeddings_32d.parquet'}")
    
    # ==================== 5. 初始化图像编码器 ====================
    print("\n[5/6] 初始化图像编码器...")
    image_encoder = ImageEncoder(output_dim=OUTPUT_DIM).to(DEVICE)
    
    # ==================== 6. 提取图像特征 (ResNet18) ====================
    print("\n[6/6] 提取图像特征 (ResNet18)...")
    image_files = sorted(IMAGE_DIR.glob("*.png"))
    print(f"  - 找到 {len(image_files)} 张图像")
    
    batch_size = 64  # 图像批次大小
    all_image_features = []
    movie_ids_from_images = []
    
    for i in tqdm(range(0, len(image_files), batch_size), desc="图像编码"):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = image_encoder.transform(img)  # (3, 224, 224)
                batch_images.append(img_tensor)
                # 从文件名提取 MovieID (例如 "1.png" -> 1)
                movie_ids_from_images.append(int(img_path.stem))
            except Exception as e:
                print(f"  - 警告: 无法加载 {img_path.name}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # 堆叠为 batch
        batch_tensor = torch.stack(batch_images).to(DEVICE)  # (batch, 3, 224, 224)
        
        # 编码 (ResNet18 输出 512 维)
        features = image_encoder(batch_tensor).cpu()  # (batch, 512)
        all_image_features.append(features)
    
    image_features = torch.cat(all_image_features, dim=0)  # (M, 512)
    print(f"  - 图像特征形状: {image_features.shape}")
    
    # ==================== 7. PCA 降维图像特征 ====================
    print(f"\n[7/7] PCA 降维图像特征 (512 -> {OUTPUT_DIM})...")
    image_encoder.pca.fit(image_features.to(DEVICE))
    image_embeddings = image_encoder.pca(image_features.to(DEVICE)).cpu()  # (M, 32)
    print(f"  - 降维后形状: {image_embeddings.shape}")
    print(f"  - 解释方差比: {image_encoder.pca.explained_variance_[:5].cpu().numpy()}")
    
    # 保存图像嵌入为 Parquet
    image_movie_ids = np.array(movie_ids_from_images)
    image_emb_np = image_embeddings.numpy()
    
    df_image_emb = pd.DataFrame({
        'movie_id': image_movie_ids,
        **{f'emb_{i}': image_emb_np[:, i] for i in range(OUTPUT_DIM)}
    })
    df_image_emb.to_parquet(
        OUTPUT_DIR / "image_embeddings_32d.parquet",
        index=False,
        compression='snappy'
    )
    print(f"  - 已保存到: {OUTPUT_DIR / 'image_embeddings_32d.parquet'}")
    
    # ==================== 8. 保存编码器 (可选) ====================
    print("\n[8/8] 保存编码器模型...")
    torch.save({
        'text_encoder': text_encoder.state_dict(),
        'image_encoder': image_encoder.state_dict()
    }, OUTPUT_DIR / "encoders.pt")
    print(f"  - 已保存到: {OUTPUT_DIR / 'encoders.pt'}")
    
    print("\n✅ 完成！所有嵌入已保存到:", OUTPUT_DIR)


    
if __name__ == "__main__":
    main()
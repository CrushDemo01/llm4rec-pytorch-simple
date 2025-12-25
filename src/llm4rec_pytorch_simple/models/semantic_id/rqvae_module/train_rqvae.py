import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json
import os
import argparse
import pickle
import torch.nn.functional as F
from llm4rec_pytorch_simple.models.semantic_id.rqvae_module.rqvae import RQVAE
from typing import Dict, Tuple, List
from collections import defaultdict


class EmbeddingLoader(Dataset):
    def __init__(self, mode: str = 'text', text_path: str = None, image_path: str = None):
        self.mode = mode
        
        # 加载数据（正确处理元组返回值）
        if text_path:
            text_df, text_mean = self._load_emb(text_path)
        else:
            text_df, text_mean = None, None
            
        if image_path:
            image_df, image_mean = self._load_emb(image_path)
        else:
            image_df, image_mean = None, None

        self.emb = None
        if mode == 'text':
            self.emb, self.emb_mean = text_df, text_mean
            # 保存 movie_ids 用于后续保存结果
            self.movie_ids = pd.read_parquet(text_path)['movie_id'].tolist()
        elif mode == 'image':
            self.emb, self.emb_mean = image_df, image_mean
            self.movie_ids = pd.read_parquet(image_path)['movie_id'].tolist()
        else:  # concat
            self.emb = torch.cat([text_df, image_df], dim=1)
            self.emb_mean = torch.cat([text_mean, image_mean], dim=0)
            self.movie_ids = pd.read_parquet(text_path)['movie_id'].tolist()

        self.length = len(self.emb)

    def _load_emb(self, emb_path: str = None):
        emb_df = pd.read_parquet(emb_path)
        # 跳过第一列（movie_id），只取 embedding 列
        emb_values = emb_df.iloc[:, 1:].values  # 跳过 movie_id
        emb_tensor = torch.tensor(emb_values, dtype=torch.float32)  # 改为 float32
        emb_mean = torch.tensor(emb_values.mean(axis=0), dtype=torch.float32)
        return emb_tensor, emb_mean
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """直接返回第 idx 行的 embedding（已跳过 movie_id 列）"""
        return self.emb[idx]

"""
语义ID映射构建工具（简化版）

从 RQVAE 训练输出的 codes 和 movie_id_mapping 构建完整的语义ID映射
移除了 Trie 树，只保留核心映射表
"""

def build_semantic_id_mapping(
    codes_df: pd.DataFrame,
    mapping_path: str
) -> dict:
    """
    从 codes DataFrame 和 movie_id_mapping 构建完整映射（简化版）

    参数:
        codes_df: 包含 'movie_id', 'code_0', 'code_1', ... 的 DataFrame
        mapping_path: movie_id_mapping.json 的路径

    返回:
        完整的映射字典，包含：
        - item_id_to_semantic_ids: {item_id: tuple(semantic_ids)}
        - semantic_ids_to_item_id: {tuple(semantic_ids): item_id}
        - num_codebooks: codebook 数量
        - vocab_size: 最大 code 值 + 1
    """
    print(f"从 {mapping_path} 加载 movie_id 映射...")
    mapping_data = json.load(open(mapping_path, 'r'))
    movie_id_to_item_id = mapping_data['movie_id_to_idx']  # movie_id (str) -> item_id (int)
    
    # 确定 codebook 数量
    code_columns = [col for col in codes_df.columns if col.startswith('code_')]
    num_codebooks = len(code_columns)
    print(f"Codebook 数量: {num_codebooks}")
    
    # 初始化映射表
    item_id_to_semantic_ids: Dict[int, Tuple[int, ...]] = {}
    semantic_ids_to_item_id: Dict[Tuple[int, ...], int] = {}

    # 记录冲突的 sid
    sid_conflict = defaultdict(int)
    conflict_count = 0
    max_code_value = 0

    print("构建语义ID映射...")
    # 遍历每一条
    for i, row in codes_df.iterrows():
        codes = row[code_columns].tolist()
        movie_id = str(row['movie_id'])

        # 转换 movie_id -> item_id
        if movie_id not in movie_id_to_item_id:
            print(f"警告: movie_id {movie_id} 不在映射表中，跳过")
            continue
        item_id = movie_id_to_item_id[movie_id]

        codes_tuple = tuple(codes)

        # 跟踪最大 code 值
        max_code_value = max(max_code_value, max(codes))

        # 获取当前 code 序列的消歧索引（所有物品都添加，确保长度一致）
        conflict_idx = sid_conflict[codes_tuple]
        sid_conflict[codes_tuple] += 1

        # 检查是否存在冲突
        if conflict_idx > 0:
            conflict_count += 1

        # 添加消歧索引（所有物品都添加）。不冲突也加，这样统一长度为 num_codebooks + 1
        codes_tuple_with_idx = tuple(codes + [conflict_idx])

        # 存储映射（使用 item_id 作为 key）
        item_id_to_semantic_ids[item_id] = codes_tuple_with_idx
        semantic_ids_to_item_id[codes_tuple_with_idx] = item_id

    # 计算 vocab_size（最大消歧索引可能大于最大 code 值）
    max_conflict_idx = max(sid_conflict.values()) - 1
    vocab_size = max(max_code_value, max_conflict_idx) + 1

    print(f"映射表构建完成:")
    print(f"  - 物品总数: {len(item_id_to_semantic_ids)}")
    print(f"  - 冲突物品数: {conflict_count}")
    print(f"  - 语义ID长度: {num_codebooks + 1}")
    print(f"  - Vocab 大小: {vocab_size}")

    return {
        'item_id_to_semantic_ids': item_id_to_semantic_ids,
        'semantic_ids_to_item_id': semantic_ids_to_item_id,
        'num_codebooks': num_codebooks,
        'vocab_size': vocab_size
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    dataset = EmbeddingLoader(
        mode=args.mode,
        text_path=args.text_path,
        image_path=args.image_path
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 获取输入维度
    sample_x = dataset[0]
    input_dim = sample_x.shape[0]
    print(f"Input dimension: {input_dim}")

    # 初始化模型
    model = RQVAE(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        code_dim=args.code_dim,
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 学习率调度器：线性衰减到 min_lr
    # 使用 LinearLR 从 lr 线性衰减到 min_lr
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1.0, 
        end_factor=args.min_lr / args.lr,
        total_iters=args.epochs
    )
    
    # 训练循环
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        total_recon_loss = 0
        total_commit_loss = 0
        total_codebook_loss = 0

        for x in dataloader:
            x = x.to(device)

            optimizer.zero_grad()
            codes, x_recon, commit_loss, codebook_loss = model(x)
            recon_loss = F.mse_loss(x_recon, x)
            beta = 0.25 + (epoch / args.epochs) * 0.25
            commit_loss *= beta
            # 总损失 = 重构损失 + beta * 承诺损失 + 码本损失
            loss = recon_loss + commit_loss + codebook_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_commit_loss += commit_loss.item()
            total_codebook_loss += codebook_loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon_loss / len(dataloader)
        avg_commit = total_commit_loss / len(dataloader)
        avg_codebook = total_codebook_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.6f}, Recon: {avg_recon:.6f}, Commit: {avg_commit:.6f}, Codebook: {avg_codebook:.6f}, LR: {current_lr:.6f}")

        # 更新学习率
        scheduler.step()

    # 保存模型
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, f"rqvae_{args.mode}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 生成并保存 codes
    model.eval()
    all_codes_list = []
    with torch.no_grad():
        eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        for x in eval_loader:
            x = x.to(device)
            codes, x_recon, commit_loss, codebook_loss = model(x)
            all_codes_list.append(codes.cpu())

    all_codes = torch.cat(all_codes_list, dim=0).numpy()

    # 保存为 Parquet
    code_cols = [f"code_{i}" for i in range(args.num_codebooks)]
    df_codes = pd.DataFrame(all_codes, columns=code_cols)
    df_codes['movie_id'] = dataset.movie_ids

    # 调整列顺序
    df_codes = df_codes[['movie_id'] + code_cols]

    codes_path = os.path.join(args.save_dir, f"rqvae_codes_{args.mode}.parquet")
    df_codes.to_parquet(codes_path)
    print(f"Codes saved to {codes_path}")
    
    # 构建并保存语义ID映射
    print("\n构建语义ID映射...")
    mapping = build_semantic_id_mapping(
        codes_df=df_codes,
        mapping_path=args.mapping_path
    )
    
    mapping_save_path = os.path.join(args.save_dir, f"semantic_id_mapping_{args.mode}.pkl")
    with open(mapping_save_path, 'wb') as f:
        pickle.dump(mapping, f)
    print(f"映射已保存到 {mapping_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path", type=str, default="/Users/yyds/workspace/REC/llm4rec-pytorch-simple/ml-1m/multimodal_datasets/embeddings_32d/text_embeddings_32d.parquet")
    parser.add_argument("--image_path", type=str, default="/Users/yyds/workspace/REC/llm4rec-pytorch-simple/ml-1m/multimodal_datasets/embeddings_32d/image_embeddings_32d.parquet")
    parser.add_argument("--mapping_path", type=str, default="/Users/yyds/workspace/REC/llm4rec-pytorch-simple/ml-1m/processed/movie_id_mapping.json")
    parser.add_argument("--mode", type=str, choices=['text', 'image', 'concat'], default='text')
    parser.add_argument("--save_dir", type=str, default="/Users/yyds/workspace/REC/llm4rec-pytorch-simple/ml-1m/multimodal_datasets/rqvae_results")

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, nargs='+', default=[512])
    parser.add_argument("--code_dim", type=int, default=50)
    parser.add_argument("--num_codebooks", type=int, default=3)
    parser.add_argument("--codebook_size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-4, help="Minimum learning rate for linear decay")
    parser.add_argument("--epochs", type=int, default=2000)
    
    args = parser.parse_args()
    train(args)

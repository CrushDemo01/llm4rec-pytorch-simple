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
from llm4rec_pytorch_simple.rqvae_module.rqvae import RQVAE
from typing import Dict, Tuple, List
from collections import defaultdict
from omegaconf import OmegaConf
import hydra
import logging

logger = logging.getLogger(__name__)


class EmbeddingLoader(Dataset):
    def __init__(self, mode: str = 'text', map_path: str = None, text_path: str = None, image_path: str = None):
        self.mode = mode
        with open(map_path, 'r') as f:
            map_dict = json.load(f)
            self.movieId2Idx = map_dict['movie_id_to_idx']
            self.idx2MovieId = map_dict['idx_to_movie_id']
            assert len(self.movieId2Idx) == len(self.idx2MovieId), \
                "movie_id_to_idx and idx_to_movie_id must have the same length"
            
        text_df, image_df = None, None
        # 加载数据（正确处理元组返回值）
        if text_path:
            text_df = pd.read_parquet(text_path)

        if image_path:
            image_df = pd.read_parquet(image_path)

        # 使用字典存储：key=movie_id, value=embedding_tensor
        self.emb_dict = {}

        if mode == 'text':
            movie_ids = text_df['movie_id'].astype(int).tolist()
            emb_values = text_df.iloc[:, 1:].values
            emb_tensors = torch.tensor(emb_values, dtype=torch.float32)
        elif mode == 'image':
            movie_ids = image_df['movie_id'].astype(int).tolist()
            emb_values = image_df.iloc[:, 1:].values
            emb_tensors = torch.tensor(emb_values, dtype=torch.float32)
        else:  # concat
            # 两个 根据 movieid join起来
            merged_df = pd.merge(text_df, image_df, on='movie_id', how='inner')
            movie_ids = merged_df['movie_id'].astype(int).tolist()
            emb_values = merged_df.iloc[:, 1:].values
            emb_tensors = torch.tensor(emb_values, dtype=torch.float32)

        # 构建字典
        for i, movie_id in enumerate(movie_ids):
            self.emb_dict[movie_id] = emb_tensors[i]
        
        # 计算所有嵌入的均值
        all_emb_tensors = torch.stack(list(self.emb_dict.values()))
        self.emb_mean = all_emb_tensors.mean(dim=0)

        # 创建索引到 movie_id 的映射列表（只包含有 embedding 的 movie_ids）
        # 按照 idx 顺序排列
        self.idx_to_movie_id_list = []
        for idx_str in sorted(self.idx2MovieId.keys(), key=lambda x: int(x)):
            movie_id = int(self.idx2MovieId[idx_str])
            if movie_id in self.emb_dict:
                self.idx_to_movie_id_list.append(movie_id)
        
        self.length = len(self.idx_to_movie_id_list)
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """直接返回第 idx 行的 embedding（已跳过 movie_id 列）"""
        movie_id = self.idx_to_movie_id_list[idx]
        return movie_id, self.emb_dict[movie_id]

    def _get_movieId_embedding(self, movieId: int):
        return self.emb_dict[movieId]

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
        - movie_id_to_semantic_ids: {movie_id: tuple(semantic_ids)}
        - num_codebooks: codebook 数量
        - vocab_size: 最大 code 值 + 1
    """
    logger.info(f"从 {mapping_path} 加载 movie_id 映射...")
    mapping_data = json.load(open(mapping_path, 'r'))
    movie_id_to_item_id = mapping_data['movie_id_to_idx']  # movie_id (str) -> item_id (int)

    # 确定 codebook 数量
    code_columns = [col for col in codes_df.columns if col.startswith('code_')]
    num_codebooks = len(code_columns)
    logger.info(f"Codebook 数量: {num_codebooks}")
    
    # 初始化映射表
    item_id_to_semantic_ids: Dict[int, Tuple[int, ...]] = {}
    semantic_ids_to_item_id: Dict[Tuple[int, ...], int] = {}
    movie_id_to_semantic_ids: Dict[str, Tuple[int, ...]] = {}
    semantic_ids_to_movie_id: Dict[Tuple[int, ...], str] = {}  # 新增：semantic_id -> movie_id 反向映射
    
    # 用于保存 CSV 的列表
    mapping_table_list = []

    # 记录冲突的 sid
    sid_conflict = defaultdict(int)
    conflict_count = 0
    max_code_value = 0

    logger.info("构建语义ID映射...")
    # 遍历每一条
    for i, row in codes_df.iterrows():
        codes = row[code_columns].tolist()
        movie_id = str(row['movie_id'])

        # 转换 movie_id -> item_id
        if movie_id not in movie_id_to_item_id:
            logger.warning(f"警告: movie_id {movie_id} 不在映射表中，跳过")
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
        movie_id_to_semantic_ids[movie_id] = codes_tuple_with_idx
        semantic_ids_to_movie_id[codes_tuple_with_idx] = movie_id  # 新增反向映射
        
        # 添加到列表用于保存 CSV
        mapping_table_list.append({
            'movie_id': movie_id,
            'idx': item_id,
            'semantic_id': str(codes_tuple_with_idx)
        })

    # 计算 vocab_size（最大消歧索引可能大于最大 code 值）
    max_conflict_idx = max(sid_conflict.values()) - 1
    vocab_size = max(max_code_value, max_conflict_idx) + 1

    logger.info(f"映射表构建完成:")
    logger.info(f"  - 物品总数: {len(item_id_to_semantic_ids)}")
    logger.info(f"  - 冲突物品数: {conflict_count}")
    logger.info(f"  - 语义ID长度: {num_codebooks + 1}")
    logger.info(f"  - Vocab 大小: {vocab_size}")

    return {
        'item_id_to_semantic_ids': item_id_to_semantic_ids,
        'semantic_ids_to_item_id': semantic_ids_to_item_id,
        'movie_id_to_semantic_ids': movie_id_to_semantic_ids,
        'semantic_ids_to_movie_id': semantic_ids_to_movie_id,  # 新增反向映射
        'num_codebooks': num_codebooks,
        'vocab_size': vocab_size,
        'mapping_table_df': pd.DataFrame(mapping_table_list)
    }


def train(cfg):
    """
    训练 RQVAE 模型
    
    参数:
        cfg: Hydra 配置对象，包含所有训练参数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 从配置中获取参数
    text_path = cfg.model.text_path
    image_path = cfg.model.image_path
    mapping_path = cfg.model.mapping_path
    mode = cfg.model.mode
    save_dir = cfg.model.save_dir
    batch_size = cfg.model.batch_size
    hidden_dim = cfg.model.hidden_dim
    code_dim = cfg.model.code_dim
    num_codebooks = cfg.model.num_codebooks
    codebook_size = cfg.model.codebook_size
    beta = cfg.model.beta
    lr = cfg.model.lr
    min_lr = cfg.model.min_lr
    epochs = cfg.model.epochs
    val_ratio = cfg.model.get('val_ratio', 0.1)  # 验证集比例，默认10%
    early_stopping_patience = cfg.model.get('early_stopping_patience', 50)  # 早停耐心值

    # 加载数据
    dataset = EmbeddingLoader(
        mode=mode,
        map_path=mapping_path,
        text_path=text_path,
        image_path=image_path
    )
    
    # 划分训练集和验证集
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # 固定随机种子确保可复现
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"数据集划分: 训练集 {train_size} 样本, 验证集 {val_size} 样本 (验证比例: {val_ratio:.2%})")

    # 获取输入维度
    _, sample_embedding = dataset[0]
    input_dim = sample_embedding.shape[0]
    logger.info(f"Input dimension: {input_dim}")

    # 初始化模型
    model = RQVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        code_dim=code_dim,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 学习率调度器：线性衰减到 min_lr
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1.0, 
        end_factor=min_lr / lr,
        total_iters=epochs
    )
    
    # 训练循环
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_commit_loss = 0
        train_codebook_loss = 0

        for batch in train_loader:
            movie_ids, x = batch
            x = x.to(device)

            optimizer.zero_grad()
            codes, x_recon, commit_loss, codebook_loss = model(x)
            recon_loss = F.mse_loss(x_recon, x)
            current_beta = beta + (epoch / epochs) * beta  # 从 beta 线性增加到 2*beta
            commit_loss *= current_beta
            # 总损失 = 重构损失 + beta * 承诺损失 + 码本损失
            loss = recon_loss + commit_loss + codebook_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_commit_loss += commit_loss.item()
            train_codebook_loss += codebook_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_recon = train_recon_loss / len(train_loader)
        avg_train_commit = train_commit_loss / len(train_loader)
        avg_train_codebook = train_codebook_loss / len(train_loader)
        
        # ===== 验证阶段 =====
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_commit_loss = 0
        val_codebook_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                movie_ids, x = batch
                x = x.to(device)
                codes, x_recon, commit_loss, codebook_loss = model(x)
                recon_loss = F.mse_loss(x_recon, x)
                loss = recon_loss + commit_loss + codebook_loss
                
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_commit_loss += commit_loss.item()
                val_codebook_loss += codebook_loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon = val_recon_loss / len(val_loader)
        avg_val_commit = val_commit_loss / len(val_loader)
        avg_val_codebook = val_codebook_loss / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}]")
            logger.info(f"  Train - Loss: {avg_train_loss:.6f}, Recon: {avg_train_recon:.6f}, Commit: {avg_train_commit:.6f}, Codebook: {avg_train_codebook:.6f}")
            logger.info(f"  Val   - Loss: {avg_val_loss:.6f}, Recon: {avg_val_recon:.6f}, Commit: {avg_val_commit:.6f}, Codebook: {avg_val_codebook:.6f}")
            logger.info(f"  LR: {current_lr:.6f}")
            
        # ===== 早停检查 =====
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            # 保存最佳模型
            os.makedirs(save_dir, exist_ok=True)
            best_model_path = os.path.join(save_dir, f"rqvae_{mode}_best.pth")
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"\n早停触发! 最佳验证损失: {best_val_loss:.6f} (Epoch {best_epoch})")
                break
        
        # 更新学习率
        scheduler.step()

    logger.info(f"\n训练完成!")
    logger.info(f"最佳验证损失: {best_val_loss:.6f} (Epoch {best_epoch})")

    # 加载最佳模型进行最终保存
    if best_epoch > 0:
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"已加载最佳模型 (Epoch {best_epoch})")

    # 保存最终模型
    os.makedirs(save_dir, exist_ok=True)
    final_model_path = os.path.join(save_dir, f"rqvae_{mode}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"模型已保存到: {save_dir}")

    # 生成并保存 codes（使用整个数据集）
    model.eval()
    all_codes_list = []
    with torch.no_grad():
        eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch in eval_loader:
            movie_ids, x = batch
            x = x.to(device)
            codes, x_recon, commit_loss, codebook_loss = model(x)
            all_codes_list.append(codes.cpu())

    all_codes = torch.cat(all_codes_list, dim=0).numpy()

    # 保存为 Parquet
    code_cols = [f"code_{i}" for i in range(num_codebooks)]
    df_codes = pd.DataFrame(all_codes, columns=code_cols)
    # 使用 idx_to_movie_id_list 获取 movie_ids，确保顺序与数据集一致
    movie_ids = dataset.idx_to_movie_id_list
    df_codes['movie_id'] = movie_ids

    # 调整列顺序
    df_codes = df_codes[['movie_id'] + code_cols]

    codes_path = os.path.join(save_dir, f"rqvae_codes_{mode}.parquet")
    df_codes.to_parquet(codes_path)
    logger.info(f"Codes saved to {codes_path}")

    # 构建并保存语义ID映射
    logger.info("\n构建语义ID映射...")
    mapping = build_semantic_id_mapping(
        codes_df=df_codes,
        mapping_path=mapping_path
    )

    # 保存映射表 CSV
    if 'mapping_table_df' in mapping:
        mapping_table_df = mapping.pop('mapping_table_df')
        mapping_table_path = os.path.join(save_dir, f"semantic_id_mapping_table_{mode}.csv")
        mapping_table_df.to_csv(mapping_table_path, index=False)
        logger.info(f"映射表已保存到 {mapping_table_path}")

    mapping_save_path = os.path.join(save_dir, f"semantic_id_mapping_{mode}.pkl")
    with open(mapping_save_path, 'wb') as f:
        pickle.dump(mapping, f)
    logger.info(f"映射已保存到 {mapping_save_path}")

    # 保存综合 JSON 映射文件（包含双向映射）
    json_mapping = {
        # 正向映射: movie_id/idx -> semantic_id
        "movie_id_to_semantic_ids": {k: list(v) for k, v in mapping['movie_id_to_semantic_ids'].items()},
        "idx_to_semantic_ids": {k: list(v) for k, v in mapping['item_id_to_semantic_ids'].items()},
        
        # 反向映射: semantic_id -> movie_id/idx
        "semantic_ids_to_movie_id": {str(list(k)): v for k, v in mapping['semantic_ids_to_movie_id'].items()},
        "semantic_ids_to_idx": {str(list(k)): v for k, v in mapping['semantic_ids_to_item_id'].items()}
    }
    json_mapping_path = os.path.join(save_dir, f"semantic_id_mapping_comprehensive_{mode}.json")
    with open(json_mapping_path, 'w') as f:
        json.dump(json_mapping, f, indent=4)
    logger.info(f"综合 JSON 映射已保存到 {json_mapping_path}")
    logger.info(f"  - 支持查询: movie_id -> semantic_id, idx -> semantic_id")
    logger.info(f"  - 支持反向查询: semantic_id -> movie_id, semantic_id -> idx")



@hydra.main(version_base="1.2", config_path="../configs/model", config_name="RQVAE.yaml")
def main(cfg):
    """
    主函数，通过 Hydra 配置启动训练
    
    参数:
        cfg: Hydra 配置对象，自动从配置文件加载
    """
    
    train(cfg)

if __name__ == "__main__":
    # 优先使用 Hydra 配置
    main()

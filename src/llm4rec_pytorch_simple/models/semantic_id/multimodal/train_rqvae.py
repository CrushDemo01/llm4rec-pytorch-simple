import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json
import os
import argparse
import torch.nn.functional as F
from llm4rec_pytorch_simple.multimodal.rqvae import RQVAE

class EmbeddingLoader(Dataset):
    def __init__(self, mode: str = 'text', text_path: str = None, image_path: str = None, mapping_path: str = None):
        self.mode = mode
        self.mapping_path = mapping_path
        
        # 1. 加载映射
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
            self.movie_id_to_idx = mapping['movie_id_to_idx']

        # 记录所有 movie_id 以便后续保存结果
        self.idx_to_movie_id = {int(v): k for k, v in self.movie_id_to_idx.items()}
        self.all_indices = sorted(self.idx_to_movie_id.keys())
        self.movie_ids = [self.idx_to_movie_id[i] for i in self.all_indices]
        
        # 2. 加载数据
        self.text_dict, self.text_dim, self.text_mean = self._load_emb(text_path) if text_path else ({}, 0, None)
        self.image_dict, self.image_dim, self.image_mean = self._load_emb(image_path) if image_path else ({}, 0, None)

    def _load_emb(self, path):
        df = pd.read_parquet(path)
        # 自动检测维度：排除 movie_id 列
        dim = len(df.columns) - 1
        # 修正映射逻辑
        df['idx'] = df['movie_id'].astype(str).map(self.movie_id_to_idx)
        df = df.dropna(subset=['idx'])  # 去除无法映射的行
        
        # 转换为字典
        data_dict = {int(row['idx']): torch.from_numpy(row.iloc[1:dim+1].values.astype('float32')) for _, row in df.iterrows()}
        
        # 计算均值用于填充
        all_vecs = torch.stack(list(data_dict.values()))
        mean_vec = all_vecs.mean(dim=0)
        
        return data_dict, dim, mean_vec

    def __len__(self):
        return len(self.all_indices)

    def __getitem__(self, idx):
        actual_idx = self.all_indices[idx]

        def get_v(d, mean_v):
            if actual_idx in d:
                return d[actual_idx]
            return mean_v

        if self.mode == 'text':
            return get_v(self.text_dict, self.text_mean)
        elif self.mode == 'image':
            return get_v(self.image_dict, self.image_mean)
        else: # concat
            t = get_v(self.text_dict, self.text_mean)
            i = get_v(self.image_dict, self.image_mean)
            return torch.cat([t, i], dim=0)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    dataset = EmbeddingLoader(
        mode=args.mode,
        text_path=args.text_path,
        image_path=args.image_path,
        mapping_path=args.mapping_path
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

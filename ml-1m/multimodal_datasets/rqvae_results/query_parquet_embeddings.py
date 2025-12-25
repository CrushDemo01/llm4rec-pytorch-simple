"""
Parquet 嵌入查询示例
展示如何从 Parquet 文件中进行 KV 查询
"""
import pandas as pd

# ==================== 方式 1: Pandas (最简单) ====================
print("=" * 60)
print("方式 1: Pandas 查询")
print("=" * 60)

# 读取
df_text = pd.read_parquet("/Users/yyds/workspace/REC/llm4rec-pytorch-simple/ml-1m/multimodal_datasets/rqvae_results/rqvae_codes_text.parquet")
df_concat = pd.read_parquet("/Users/yyds/workspace/REC/llm4rec-pytorch-simple/ml-1m/multimodal_datasets/rqvae_results/rqvae_codes_concat.parquet")

print(f"文本嵌入形状: {df_text.shape}")
print(f"concat 嵌入形状: {df_concat.shape}")
print(f"\n列名: {df_text.columns.tolist()[:5]}...")  # 前5列
print(df_text.head())
print(df_concat.head())
# 最大id
print(df_text['movie_id'])
print(df_concat['movie_id'])

# KV 查询：根据 movie_id 查询
movie_id = 1
text_emb = df_text[df_text['movie_id'] == movie_id].iloc[0]
concat_emb = df_concat[df_concat['movie_id'] == movie_id].iloc[0]
print(text_emb.shape)
print(concat_emb.shape)
print(f"\nMovie {movie_id} 的文本嵌入 (前5维): {text_emb[1:6].values}")
print(f"\nMovie {movie_id} 的 concat 嵌入 (前5维): {concat_emb[1:6].values}")

# 批量查询
movie_ids = [1, 2, 3]
batch_emb = df_text[df_text['movie_id'].isin(movie_ids)]
print(f"\n批量查询 {movie_ids}:")
print(batch_emb[['movie_id', 'emb_0', 'emb_1', 'emb_2']])


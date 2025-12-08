import torch

# 创建原始 embedding
embedding = torch.nn.Embedding(5, 3)
print("初始 embedding.weight:")
print(embedding.weight.data)

# 方式1: 使用 .data（你之前的方式）
embeddings_t_data = embedding.weight.t()

print("\n使用 .data 创建的 embeddings_t:")
print(embeddings_t_data)

# 现在更新 embedding 的权重（模拟训练）
with torch.no_grad():
    embedding.weight[0] = torch.tensor([999., 999., 999.])

print("\n更新后的 embedding.weight:")
print(embedding.weight.data)

print("\nembeddings_t_data 是否同步更新？")
print(embeddings_t_data)
# ❌ 不会！因为 .t() 创建了新的内存
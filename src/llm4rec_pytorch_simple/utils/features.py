import torch


def seq_features_from_row(
    batch,
    device: torch.device,
    max_output_length: int,
):
    # batch中包含的全部特征在这里
    historical_lengths = batch["history_lengths"]  # [B]
    historical_ids = batch["historical_ids"]  # [B, N]
    historical_ratings = batch["historical_ratings"]
    historical_timestamps = batch["historical_timestamps"]
    user_id = batch["user_id"]
    target_ids = batch["target_ids"].unsqueeze(1)  # [B, 1]
    target_ratings = batch["target_ratings"].unsqueeze(1)
    target_timestamps = batch["target_timestamps"].unsqueeze(1)

    B = historical_lengths.size(0)  # 批次大小
    # 将output结果序列，cat到现有历史序列后面
    zero_output_ids = torch.zeros((B, max_output_length), dtype=historical_ids.dtype, device=device)  # 零填充部分
    historical_ids = torch.cat([historical_ids, zero_output_ids], dim=1)
    # 同样的，对历史评分序列也进行零填充
    historical_ratings = torch.cat(
        [
            historical_ratings,  # 原始评分
            torch.zeros(
                (B, max_output_length),  # 零填充部分
                dtype=historical_ratings.dtype,
                device=device,
            ),
        ],
        dim=1,
    )
    # 同样的，对历史时间戳序列也进行零填充
    historical_timestamps = torch.cat(
        [
            historical_timestamps,  # 原始时间戳
            torch.zeros(
                (B, max_output_length),  # 零填充部分
                dtype=historical_timestamps.dtype,
                device=device,
            ),
        ],
        dim=1,
    )

    # 特殊处理：在真实历史长度位置插入目标时间戳
    # 这样可以标记历史序列的结束位置和目标物品的时间关系
    """
    # 原始数据（批次大小B=2）
    historical_lengths = [3, 2]  # 用户1有3个历史，用户2有2个历史
    historical_timestamps = [
        [100, 200, 300, 0, 0],    # 用户1：3个真实时间戳 + 2个填充0
        [150, 250, 0, 0, 0]       # 用户2：2个真实时间戳 + 3个填充0
    ]
    target_timestamps = [400, 350]  # 目标物品时间戳

    # scatter_操作后
    historical_timestamps = [
        [100, 200, 300, 400, 0],  # 400插入到索引3位置（historical_lengths[0]=3）
        [150, 250, 350, 0, 0]     # 350插入到索引2位置（historical_lengths[1]=2）
    ]
    """
    historical_timestamps.scatter_(
        dim=1,
        index=historical_lengths.view(-1, 1),  # 在真实长度位置
        src=target_timestamps.view(-1, 1),  # 插入目标时间戳
    )
    # 返回两个字典，第一个字典包含历史序列，第二个字典包含目标序列
    return {
        "user_id": user_id,
        "historical_lengths": historical_lengths,
        "historical_ids": historical_ids,
        "historical_ratings": historical_ratings,
        "historical_timestamps": historical_timestamps,
    }, {
        "target_ids": target_ids,
        "target_ratings": target_ratings,
        "target_timestamps": target_timestamps,
    }


def get_current_embeddings_simple(
    lengths: torch.Tensor,
    encoded_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    获取序列的当前嵌入向量，通常是"最后一个有效位置"的嵌入

    在推荐系统中，常用于从编码后的序列中提取用户的当前状态表示，
    例如从用户历史行为序列中提取最新的行为特征用于预测。

    参数:
        lengths: (B,) 形状的整数张量，表示每个序列的实际长度。
        encoded_embeddings: (B, N, D) 形状的浮点张量，编码后的序列嵌入。

    返回:
        (B, D) 形状的浮点张量，每个序列的最后一个有效位置的嵌入。
                即[i, :]等于encoded_embeddings[i, lengths[i] - 1, :]
    """
    B = encoded_embeddings.size(0)
    # 批次索引
    batch_indices = torch.arange(B, dtype=lengths.dtype, device=lengths.device)
    # 每个序列的最后位置
    last_positions = lengths - 1
    return encoded_embeddings[batch_indices, last_positions, :]

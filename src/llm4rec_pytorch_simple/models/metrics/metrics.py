import torch


class RetrievalMetrics:
    """
    原生 PyTorch 实现的检索指标计算器。
    不依赖 torchmetrics，适用于单机环境。

    功能：
    - 累积(Accumulate)每个 batch 的预测结果
    - 计算(Compute) NDCG@K, HR@K, MRR
    - 修正了 MRR 在未命中时的计算逻辑
    """

    def __init__(self, topk: int, at_k_list: list[int] = None):
        """
        Args:
            topk (int): 截断长度 (Top-K)
            at_k_list (list): 需要计算具体指标的 K 值列表 (e.g. [5, 10])
                              如果不填，默认只计算 @k
        """
        self.topk = topk
        self.at_k_list = at_k_list if at_k_list else [topk]

        # 内部状态：用于暂存每个 batch 的数据
        self.preds_buffer = []
        self.targets_buffer = []

    def reset(self):
        """重置状态，通常在每个 Epoch 开始前调用"""
        self.preds_buffer = []
        self.targets_buffer = []

    def update(self, top_k_ids: torch.Tensor, target_ids: torch.Tensor):
        """
        更新当前 batch 的结果。建议传入 detach() 后的 tensor 以节省显存。

        Args:
            top_k_ids: [Batch, K] 模型预测的 Top-K 物品 ID
            target_ids: [Batch, 1] 或 [Batch] 真实物品 ID
        """
        # 确保 target_ids 是 [Batch, 1] 形状
        if target_ids.dim() == 1:
            target_ids = target_ids.view(-1, 1)

        # 存入 buffer (移动到 CPU 以防止显存爆炸，如果显存够大可以不移)
        self.preds_buffer.append(top_k_ids.detach().cpu())
        self.targets_buffer.append(target_ids.detach().cpu())

    def compute(self):
        """
        计算所有指标。
        """
        if not self.preds_buffer:
            return {}
        # 1. 拼接所有 Batch 的数据, 当前全部 batch 的数据合并起来才是正确的一个
        all_preds = torch.cat(self.preds_buffer, dim=0)
        all_targets = torch.cat(self.targets_buffer, dim=0)

        # 2. 计算排名 (Ranking Logic)
        # 技巧：把 target 拼接到 pred 的最后一位。
        # 如果 target 在 pred 中出现，argmax 会找到它在 pred 中的位置。
        # 如果没出现，argmax 会找到最后一位 (即 index = k)。
        # extended: [Total_Samples, K+1]
        """
        # all_preds = [10, 25, 8, 15, 3]  # Top-5预测
        # all_targets = [8]                # 真实目标物品ID是8

        extended = [10, 25, 8, 15, 3, 8]  # 拼接后
        # extended == all_targets:(广播)
        # [False, False, True, False, False, True]
        #    0      1     2     3     4     5
        """
        extended = torch.cat([all_preds, all_targets], dim=1)
        # 找到 target 所在的索引 (0-based)
        # value 是 bool (是否匹配)，indices 是位置。不在Top-K中 ： argmax 会返回最后一个位置(K+1)
        _, rank_indices = torch.max(extended == all_targets, dim=1)
        # 转为 1-based rank
        ranks = rank_indices + 1

        # 3. 计算指标
        metrics = {}
        # 为了向量化计算，把 rank 转为 float
        ranks_float = ranks.float()

        # --- HR@K & NDCG@K ---
        for at_k in self.at_k_list:
            # HR: 只要 rank <= at_k 就是 1，否则 0
            hr_tensor = (ranks <= at_k).float()
            metrics[f"hr@{at_k}"] = hr_tensor.float().mean().item()

            # NDCG: 1 / log2(rank + 1)
            ndcg_tensor = torch.where(ranks <= at_k, 1.0 / torch.log2(ranks_float + 1.0), torch.zeros_like(ranks_float))
            metrics[f"ndcg@{at_k}"] = ndcg_tensor.float().mean().item()

        # --- MRR ---
        # 修正 MRR：计算所有样本的倒数排名平均值
        # 命中的样本：倒数排名 = 1/rank
        # 未命中的样本：倒数排名 = 0
        reciprocal_ranks = torch.where(
            ranks <= self.topk,
            1.0 / ranks_float,  # 命中的样本：1/rank
            torch.zeros_like(ranks_float),  # 未命中的样本：0
        )
        metrics["mrr"] = reciprocal_ranks.float().mean().item()

        return metrics


class RankingMetrics:
    """
    纯 PyTorch 手动实现的精排指标计算器。
    不依赖 sklearn 或 numpy，完全独立。

    功能：
    - LogLoss (手动实现 BCE)
    - AUC (手动实现秩和公式)
    - GAUC (基于 Torch 操作的分组计算)
    - NE (Normalized Entropy): 归一化交叉熵
    - ECE (Expected Calibration Error): 期望校准误差 (衡量 Calibration)
    """

    def __init__(self):
        self.probs_buffer = []
        self.labels_buffer = []
        self.group_ids_buffer = []

    def reset(self):
        """重置状态"""
        self.probs_buffer = []
        self.labels_buffer = []
        self.group_ids_buffer = []

    def update(self, probs: torch.Tensor, labels: torch.Tensor, group_ids: torch.Tensor = None):
        """
        累积 Batch 数据。
        """
        # 保持在 CPU 以节省显存，计算时再根据需求处理
        self.probs_buffer.append(probs.detach().cpu().view(-1))
        self.labels_buffer.append(labels.detach().cpu().view(-1))

        if group_ids is not None:
            self.group_ids_buffer.append(group_ids.detach().cpu().view(-1))

    def compute(self):
        """
        计算所有指标，返回字典。
        """
        if not self.probs_buffer:
            return {}

        # 1. 拼接数据
        all_probs = torch.cat(self.probs_buffer)
        all_labels = torch.cat(self.labels_buffer)

        metrics = {}

        # --- 2. LogLoss (Binary Cross Entropy) ---
        metrics["logloss"] = self._manual_log_loss(all_probs, all_labels)

        # --- 3. AUC (Global) ---
        metrics["auc"] = self._manual_auc(all_probs, all_labels)

        # --- 4. GAUC (Group AUC) ---
        if self.group_ids_buffer:
            all_groups = torch.cat(self.group_ids_buffer)
            metrics["gauc"] = self._calculate_gauc_torch(all_probs, all_labels, all_groups)

        # --- 5. NE (Normalized Entropy) ---
        # 依赖于刚刚算出的 logloss / background CTR的熵
        metrics["ne"] = self._manual_ne(metrics["logloss"], all_labels)

        # --- 6. Calibration (ECE) ---
        metrics["calibration"] = self._manual_ece(all_probs, all_labels, n_bins=10)

        return metrics

    def _manual_log_loss(self, probs, labels):
        """
        手动实现 LogLoss (BCE)。
        公式: -1/N * sum(y*log(p) + (1-y)*log(1-p))
        """
        # 加上极小值 epsilon 防止 log(0) -> nan
        epsilon = 1e-7
        # 限制概率范围在 [epsilon, 1-epsilon]
        probs_clipped = torch.clamp(probs, epsilon, 1.0 - epsilon)

        loss = -(labels * torch.log(probs_clipped) + (1.0 - labels) * torch.log(1.0 - probs_clipped))
        return loss.mean().item()

    def _manual_ne(self, model_logloss, labels):
        """
        计算 NE (Normalized Entropy)。样本的交叉熵均值和背景CTR的交叉熵的比值。
        - 背景CTR指的是训练样本集样本的经验CTR，可以理解成平均的点击率。但是这里要注意，不是正负样本的比例（因为我们在训练模型之前都会做采样）。
        - 除以了background CTR的熵，使得NE对background CTR不敏感
        NE = 预测的log loss / background CTR的熵
        - Background_LogLoss: 假设预测值永远是数据集的平均 CTR 时的 LogLoss。
        含义：归一化后的 LogLoss。它回答了“你的模型比盲猜（只预测平均点击率）强多少？”
        解读：NE 越小越好。如果 NE=1，说明模型和瞎猜一样；NE < 1 说明有提升。这是跨场景对比模型能力的硬通货（因为它消除了不同场景 CTR 绝对值差异的影响）。
        """
        avg_ctr = labels.float().mean().item()  # 确保labels是浮点类型，避免mean()报错

        # 边界处理：如果数据全是正样本或全是负样本，Entropy 为 0，NE 无意义
        if avg_ctr <= 1e-7 or avg_ctr >= 1.0 - 1e-7:
            return 1.0

        # 计算 Background LogLoss (熵)
        # 此时 p 是一个常数 (avg_ctr)
        # entropy = - (y * log(p_avg) + (1-y) * log(1-p_avg))
        # 由于是对所有样本求均值，且 sum(y)/N = avg_ctr
        # 公式可简化为: - (avg_ctr * log(avg_ctr) + (1-avg_ctr) * log(1-avg_ctr))
        epsilon = 1e-7
        p = max(min(avg_ctr, 1.0 - epsilon), epsilon)
        background_logloss = -(avg_ctr * torch.log(torch.tensor(p)) + (1 - avg_ctr) * torch.log(torch.tensor(1 - p)))

        return model_logloss / background_logloss.item()

    def _manual_auc(self, probs, labels):
        """
        手动实现 AUC (使用 Rank Sum 公式)。
        AUC = (正样本的Rank和 - M*(M+1)/2) / (M*N)
        M: 正样本数, N: 负样本数

        例子说明：
        假设有6个样本，预测概率和真实标签如下：

        样本索引:   0    1    2    3    4    5
        预测概率:  0.1  0.8  0.3  0.9  0.2  0.7
        真实标签:   0    1    0    1    0    1    (1=正样本，0=负样本)

        计算步骤：
        1. 按概率从小到大排序：
           排序后索引: [4, 0, 2, 1, 5, 3]
           排序后概率: [0.2, 0.1, 0.3, 0.8, 0.7, 0.9]
           排序后标签: [0, 0, 0, 1, 1, 1]

        2. 统计样本数：
           正样本数 M = 3 (索引3,4,5)
           负样本数 N = 3 (索引0,1,2)

        3. 计算正样本的Rank和：
           正样本位置: 4, 5, 6 (1-based排名)
           Rank和 = 4 + 5 + 6 = 15

        4. 应用公式：
           AUC = (15 - 3*(3+1)/2) / (3*3)
               = (15 - 6) / 9
               = 9 / 9 = 1.0

        结果解释：AUC=1.0表示模型完美区分正负样本，所有正样本的排名都高于负样本。

        如果AUC=0.5，表示模型随机猜测，无法区分正负样本。
        如果AUC=0.0，表示模型完全反向，所有正样本排名都低于负样本。
        """
        # 1. 按分数从小到大排序 (Rank Sum 需要从小到大的 Rank)
        # sort_indices: 排序后的索引
        _, sort_indices = torch.sort(probs, descending=False)
        sorted_labels = labels[sort_indices]

        # 2. 统计正负样本数
        n_pos = sorted_labels.sum()
        n_neg = len(labels) - n_pos

        # 边界检查：如果只有一类样本，无法计算 AUC
        if n_pos == 0 or n_neg == 0:
            return 0.5

        # 3. 获取 Rank
        # 排名从 1 开始 (1, 2, 3, ...)
        ranks = torch.arange(1, len(labels) + 1, device=labels.device, dtype=torch.float32)

        # 4. 提取正样本的 Rank 并求和
        # sorted_labels == 1 的位置就是正样本在有序列表中的位置
        pos_rank_sum = torch.sum(ranks[sorted_labels == 1])

        # 5. 应用公式
        # AUC = (Sum(R_pos) - M(M+1)/2) / (M*N)
        auc_score = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

        return auc_score.item()

    def _calculate_gauc_torch(self, probs, labels, group_ids):
        """
        纯 Torch 实现的 GAUC。
        逻辑：按 User ID 排序 -> 使用 unique_consecutive 切分 -> 循环计算 AUC -> 加权平均
        """
        # 1. 按 group_ids 排序，把同一个用户的样本聚在一起
        sorted_indices = torch.argsort(group_ids)
        group_ids = group_ids[sorted_indices]
        probs = probs[sorted_indices]
        labels = labels[sorted_indices]

        # 2. 找到切分点
        # unique_consecutive 返回去重后的 ID 和每个 ID 出现的次数 (counts)
        # 例如: groups=[A, A, B, B, B] -> counts=[2, 3]
        _, counts = torch.unique_consecutive(group_ids, return_counts=True)

        # 3. 切分数据 (Split)
        # split_probs 是一个 tuple，包含每个用户的 prob tensor
        split_probs = torch.split(probs, counts.tolist())
        split_labels = torch.split(labels, counts.tolist())

        auc_sum = 0.0
        weight_sum = 0.0

        # 4. 循环计算每个用户的 AUC
        # 这里无法向量化，因为每个用户的样本长度不一样
        for u_probs, u_labels in zip(split_probs, split_labels):
            # 必须既有正样本又有负样本才能算 AUC
            # 快速检查：如果 sum == 0 (全负) 或 sum == len (全正)，跳过
            u_pos_count = u_labels.sum()
            if u_pos_count > 0 and u_pos_count < len(u_labels):
                # 调用复用上面的手动 AUC
                auc = self._manual_auc(u_probs, u_labels)

                # 权重通常是该用户的样本数 (展示次数)
                weight = len(u_labels)

                auc_sum += auc * weight
                weight_sum += weight

        if weight_sum > 0:
            return auc_sum / weight_sum
        else:
            return 0.5

    def _manual_ece(self, probs, labels, n_bins=10):
        """
        计算 ECE (Expected Calibration Error)。
        将预测概率分成 n_bins 个桶，计算每个桶内的 (平均预测概率 - 真实正样本比例) 的加权平均误差。
        """
        # 1. 确定分桶边界 (0.0, 0.1, ..., 1.0)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

        # 2. 确定每个样本属于哪个桶
        # bucketize 返回索引 [1, n_bins]，我们需要 0-based index，所以减 1
        # 但 bucketize 的边界处理有点 tricky，简单起见我们手动循环处理更稳健

        ece = 0.0
        total_samples = len(probs)

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # 找到落在当前桶内的样本
            if i == n_bins - 1:
                # 最后一个桶包含 1.0
                in_bin = (probs >= bin_lower) & (probs <= bin_upper)
            else:
                # 前面的桶左闭右开
                in_bin = (probs >= bin_lower) & (probs < bin_upper)

            bin_count = in_bin.sum().item()

            if bin_count > 0:
                # 桶内平均预测概率 (Confidence)
                avg_prob = probs[in_bin].float().mean().item()
                # 桶内真实正样本比例 (Accuracy)
                avg_label = labels[in_bin].float().mean().item()

                # 加权误差: (该桶样本数/总数) * |Conf - Acc|
                ece += (bin_count / total_samples) * abs(avg_prob - avg_label)

        return ece


def test_ranking_metrics():
    """
    RankingMetrics 类的全面测试用例
    测试各种场景下的指标计算正确性
    """
    print("=== RankingMetrics 测试开始 ===")

    # 测试1：完美预测场景
    print("\n1. 完美预测测试")
    ranking_metrics = RankingMetrics()

    # 正样本高概率，负样本低概率
    perfect_probs = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    perfect_labels = torch.tensor([1, 0, 1, 0, 1, 0])

    ranking_metrics.update(perfect_probs, perfect_labels)
    perfect_results = ranking_metrics.compute()

    print(f"预测概率: {perfect_probs.tolist()}")
    print(f"真实标签: {perfect_labels.tolist()}")
    print(f"AUC: {perfect_results['auc']:.4f} (期望: 1.0)")
    print(f"LogLoss: {perfect_results['logloss']:.4f}")
    print(f"NE: {perfect_results['ne']:.4f}")

    assert abs(perfect_results["auc"] - 1.0) < 0.001, "完美预测AUC应该接近1.0"

    # 测试2：随机预测场景
    print("\n2. 随机预测测试")
    ranking_metrics.reset()

    # 概率接近0.5，无法区分正负样本
    random_probs = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    random_labels = torch.tensor([1, 0, 1, 0, 1, 0])

    ranking_metrics.update(random_probs, random_labels)
    random_results = ranking_metrics.compute()

    print(f"预测概率: {random_probs.tolist()}")
    print(f"真实标签: {random_labels.tolist()}")
    print(f"AUC: {random_results['auc']:.4f} (期望: 0.5)")
    print(f"LogLoss: {random_results['logloss']:.4f}")
    print(f"NE: {random_results['ne']:.4f} (期望: 1.0)")

    # 测试3：反向预测场景
    print("\n3. 反向预测测试")
    ranking_metrics.reset()

    # 正样本低概率，负样本高概率
    reverse_probs = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    reverse_labels = torch.tensor([1, 0, 1, 0, 1, 0])

    ranking_metrics.update(reverse_probs, reverse_labels)
    reverse_results = ranking_metrics.compute()

    print(f"预测概率: {reverse_probs.tolist()}")
    print(f"真实标签: {reverse_labels.tolist()}")
    print(f"AUC: {reverse_results['auc']:.4f} (期望: 0.0)")
    print(f"LogLoss: {reverse_results['logloss']:.4f}")

    assert reverse_results["auc"] < 0.1, "反向预测AUC应该接近0.0"

    # 测试4：边界情况测试
    print("\n4. 边界情况测试")

    # 4.1 全正样本
    ranking_metrics.reset()
    all_pos_probs = torch.tensor([0.6, 0.7, 0.8, 0.9])
    all_pos_labels = torch.tensor([1, 1, 1, 1])

    ranking_metrics.update(all_pos_probs, all_pos_labels)
    all_pos_results = ranking_metrics.compute()

    print(f"全正样本 - AUC: {all_pos_results['auc']:.4f} (期望: 0.5)")
    assert all_pos_results["auc"] == 0.5, "全正样本AUC应该是0.5"

    # 4.2 全负样本
    ranking_metrics.reset()
    all_neg_probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    all_neg_labels = torch.tensor([0, 0, 0, 0])

    ranking_metrics.update(all_neg_probs, all_neg_labels)
    all_neg_results = ranking_metrics.compute()

    print(f"全负样本 - AUC: {all_neg_results['auc']:.4f} (期望: 0.5)")
    assert all_neg_results["auc"] == 0.5, "全负样本AUC应该是0.5"

    # 测试5：GAUC测试（分组AUC）
    print("\n5. GAUC分组测试")
    ranking_metrics.reset()

    # 创建分组数据，每个用户内部有正负样本
    gauc_probs = torch.tensor([0.8, 0.2, 0.9, 0.1, 0.7, 0.3])
    gauc_labels = torch.tensor([1, 0, 1, 0, 1, 0])
    gauc_groups = torch.tensor([0, 0, 1, 1, 2, 2])  # 3个用户，每个用户2个样本

    ranking_metrics.update(gauc_probs, gauc_labels, gauc_groups)
    gauc_results = ranking_metrics.compute()

    print(f"分组预测 - AUC: {gauc_results['auc']:.4f}")
    print(f"分组预测 - GAUC: {gauc_results['gauc']:.4f}")

    # 每个用户内部都是完美预测，所以GAUC应该是1.0
    assert abs(gauc_results["gauc"] - 1.0) < 0.001, "分组完美预测GAUC应该接近1.0"

    # 测试6：多批次累积测试
    print("\n6. 多批次累积测试")
    ranking_metrics.reset()

    # 分3个批次添加数据
    batch1_probs = torch.tensor([0.8, 0.2, 0.7])
    batch1_labels = torch.tensor([1, 0, 1])
    ranking_metrics.update(batch1_probs, batch1_labels)

    batch2_probs = torch.tensor([0.9, 0.1, 0.6])
    batch2_labels = torch.tensor([1, 0, 1])
    ranking_metrics.update(batch2_probs, batch2_labels)

    batch3_probs = torch.tensor([0.5, 0.4, 0.8])
    batch3_labels = torch.tensor([1, 0, 1])
    ranking_metrics.update(batch3_probs, batch3_labels)

    multi_batch_results = ranking_metrics.compute()

    print(f"多批次累积 - AUC: {multi_batch_results['auc']:.4f}")
    print("总样本数: 9")

    # 验证累积效果
    expected_auc = RankingMetrics()._manual_auc(
        torch.cat([batch1_probs, batch2_probs, batch3_probs]), torch.cat([batch1_labels, batch2_labels, batch3_labels])
    )
    assert abs(multi_batch_results["auc"] - expected_auc) < 0.001, "多批次累积结果应该一致"

    # 测试7：ECE校准测试
    print("\n7. 校准测试 (ECE)")
    ranking_metrics.reset()

    # 创建校准较好的预测
    calibrated_probs = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    calibrated_labels = torch.tensor([0, 0, 1, 1, 1])  # 概率与标签匹配较好

    ranking_metrics.update(calibrated_probs, calibrated_labels)
    calibrated_results = ranking_metrics.compute()

    print(f"校准较好 - ECE: {calibrated_results['calibration']:.4f} (期望: 较小值)")

    print("\n=== 所有测试通过！ ===")


if __name__ == "__main__":
    # 运行测试
    test_ranking_metrics()

    # 原有示例
    print("\n=== 原有示例 ===")
    top_k_ids = torch.tensor([[10, 25, 8, 15, 3], [5, 12, 8, 20, 1]])
    target_ids = torch.tensor([8, 12])

    metrics_calc = RetrievalMetrics(topk=5, at_k_list=[5, 10])
    metrics_calc.update(top_k_ids, target_ids)
    print(metrics_calc.compute())

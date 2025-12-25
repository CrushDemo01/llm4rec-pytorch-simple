"""
TIGER 训练和验证模块

基于 Semantic ID 的生成式推荐训练模块
参考 TIGER 论文: Recommender Systems with Generative Retrieval (NeurIPS 2023)
"""
from llm4rec_pytorch_simple.models.RQVAE_recommender import RQVAERecommender

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from typing import Optional, Dict, Any, Tuple, List
from omegaconf import DictConfig

from llm4rec_pytorch_simple.models.archs.TIGER import TIGER
from llm4rec_pytorch_simple.utils.pylogger import RankedLogger
from llm4rec_pytorch_simple.utils.features import seq_features_from_row

logger = RankedLogger(__name__)


class TIGERRetrievalModule(RQVAERecommender):
    """
    TIGER 检索模块 - 基于 TIGER 模型的生成式推荐器

    需要配置的子模块：
    1. sequence_model: TIGER 模型 (encoder-decoder)
    2. semantic_id_manager: SemanticIdTokenizer (item_id ↔ semantic_id 映射)
    3. sid_embedding: Embedding 层 (semantic_id → embedding)
    4. user_embedding: 用户 ID embedding (可选)
    5. position_emb: 位置编码 (可选，TIGER 内部处理)
    """

    def __hydra_init_submodules(
        self,
        negative_sampler: DictConfig,
        sequence_model: DictConfig,
        loss: DictConfig,
        metrics: DictConfig,
        submodules
    ) -> None:
        # 调用父类初始化标准子模块
        super().__hydra_init_submodules(
            negative_sampler, sequence_model, loss, metrics, submodules
        )

        # TIGER 特有的配置参数
        self.codebook_size = submodules.get('codebook_size')
        self.id_length = submodules.get('id_length')
        self.pad_token_id = submodules.get('pad_token_id', 0)

        # TIGER 特有的子模块
        # semantic_id_manager: item_id ↔ semantic_id 映射
        sem_manager = submodules.get('semantic_id_manager')
        if isinstance(sem_manager, DictConfig):
            self.semantic_id_manager = hydra.utils.instantiate(sem_manager)
        else:
            self.semantic_id_manager = sem_manager

        # sid_embedding: semantic ID → embedding
        sid_emb = submodules.get('sid_embedding')
        if isinstance(sid_emb, DictConfig):
            self.sid_embedding = hydra.utils.instantiate(sid_emb)
        elif sid_emb is not None:
            self.sid_embedding = sid_emb
        else:
            # 尝试从 TIGER 模型获取（如果 TIGER 内部有共享权重）
            self.sid_embedding = None
            logger.warning("未提供 sid_embedding，将尝试从 TIGER 内部获取")

        # user_embedding: 用户 ID → embedding (可选)
        user_emb = submodules.get('user_embedding')
        if isinstance(user_emb, DictConfig):
            self.user_embedding = hydra.utils.instantiate(user_emb)
        elif user_emb is not None:
            self.user_embedding = user_emb
        else:
            self.user_embedding = None

        # 验证必需的子模块
        if self.semantic_id_manager is None:
            raise ValueError("必须提供 semantic_id_manager")
        if self.sid_embedding is None and not hasattr(self.sequence_model, 'in_proj'):
            raise ValueError("必须提供 sid_embedding 或 TIGER 有 in_proj 方法可用")

        # 日志
        logger.info(f"TIGERRetrievalModule 初始化完成")
        logger.info(f"  - codebook_size: {self.codebook_size}")
        logger.info(f"  - id_length: {self.id_length}")
        logger.info(f"  - semantic_id_manager: {type(self.semantic_id_manager).__name__}")

    def setup(self, stage: str) -> None:
        """Lightning setup hook - called before training/validation/testing starts"""
        # Call parent setup first (handles torch.compile if needed)
        super().setup(stage)

        # 将 sid_embedding 传递给 TIGER 模型（用于 generate() 方法）
        with open('/tmp/tiger_setup_debug.log', 'w') as f:
            f.write(f"TIGERRetrievalModule.setup() called with stage={stage}\n")
            f.write(f"检查 TIGER 模型是否有 sid_embedding 属性: {hasattr(self.sequence_model, 'sid_embedding')}\n")
            f.write(f"self.sid_embedding type: {type(self.sid_embedding)}\n")
            f.write(f"self.sequence_model type: {type(self.sequence_model)}\n")

        if hasattr(self.sequence_model, 'sid_embedding'):
            self.sequence_model.sid_embedding = self.sid_embedding
            logger.info(f"✓ sid_embedding 已传递给 TIGER 模型 (验证: {self.sequence_model.sid_embedding is not None})")
        else:
            logger.warning("⚠ TIGER 模型没有 sid_embedding 属性，无法传递")

    def _get_user_emb(self, user_ids: torch.Tensor) -> torch.Tensor:
        """获取用户 embedding，[B, 1, D]"""
        if self.user_embedding is not None:
            return self.user_embedding(user_ids).unsqueeze(1)
        else:
            # 没有用户 embedding，返回零向量（与 sid_embedding_dim 一致）
            B = user_ids.shape[0]
            D = self.item_embedding_dim  # 使用 sid_embedding_dim
            return torch.zeros(B, 1, D, device=self.device)

    def _get_semantic_id_embeddings(self, item_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 item_ids 转换为 semantic_id 序列和对应的 embeddings

        Args:
            item_ids: [B, N] 物品 ID 序列

        Returns:
            semantic_ids: [B, N, K]  semantic ID 序列
            semantic_embeddings: [B, N, K, D]  semantic ID embeddings
        """
        B, N = item_ids.shape
        device = self.device

        # 存储结果
        all_semantic_ids = []
        all_semantic_embeddings = []

        for b in range(B):
            # 获取该样本的有效序列
            valid_mask = item_ids[b] != 0
            valid_item_ids = item_ids[b][valid_mask].tolist()

            if len(valid_item_ids) == 0:
                # 没有有效物品
                sem_ids = torch.zeros(N, self.id_length, dtype=torch.long, device=device)
                sem_emb = torch.zeros(N, self.id_length, self.sid_embedding.embedding_dim, device=device)
            else:
                # 转换为 semantic_ids
                sem_ids_batch = self.semantic_id_manager.tokenize(valid_item_ids).to(device)  # [valid_len, K]

                # 处理 -1（缺失的 item），将其替换为 0（padding）
                sem_ids_batch = torch.where(sem_ids_batch == -1, torch.zeros_like(sem_ids_batch), sem_ids_batch)

                # 获取 embeddings
                # 展平后查找 embedding，然后重新调整形状
                flat_sem_ids = sem_ids_batch.view(-1)  # [valid_len * K]
                flat_emb = self.sid_embedding(flat_sem_ids)  # [valid_len * K, D]
                sem_emb_batch = flat_emb.view(len(valid_item_ids), self.id_length, -1)  # [valid_len, K, D]

                # 填充回原始长度
                sem_ids = torch.zeros(N, self.id_length, dtype=torch.long, device=device)
                sem_emb = torch.zeros(N, self.id_length, self.sid_embedding.embedding_dim, device=device)
                sem_ids[valid_mask] = sem_ids_batch
                sem_emb[valid_mask] = sem_emb_batch

            all_semantic_ids.append(sem_ids)
            all_semantic_embeddings.append(sem_emb)

        # 转换为 batch tensor
        semantic_ids = torch.stack(all_semantic_ids, dim=0)  # [B, N, K]
        semantic_embeddings = torch.stack(all_semantic_embeddings, dim=0)  # [B, N, K, D]

        return semantic_ids, semantic_embeddings

    def _build_tiger_inputs(
        self,
        seq_features: dict[str, torch.Tensor],
        target_semantic_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        构建 TIGER 模型所需的输入

        TIGER.forward() 需要:
        - user_emb: [B, 1, D]
        - post_sid_emb: [B, N, D] (历史序列 embeddings，已展平)
        - post_sid: [B, N] (历史序列 semantic IDs，用于 mask)
        - target_sid_emb: [B, K, D] (目标序列 embeddings)
        - target_padding_mask: [B, K]
        - position_emb: [B, N, D] (位置编码)

        这里使用一个简化的约定：
        - 将历史序列展平: [B, N, K, D] -> [B, N*K, D]
        - semantic IDs 也展平: [B, N, K] -> [B, N*K]
        """
        B = seq_features["historical_ids"].shape[0]
        device = self.device

        # 1. 用户 embedding
        user_emb = self._get_user_emb(seq_features["user_id"])

        # 2. 历史序列处理
        # 获取历史序列的 semantic IDs 和 embeddings
        # 注意：需要处理 historical_ids 的 padding
        historical_ids = seq_features["historical_ids"]
        historical_lengths = seq_features["historical_lengths"]

        # 展平历史序列，用于 TIGER
        # TIGER 的设计假设输入是展平的 semantic ID 序列
        # 每个历史 item 产生 K 个 semantic tokens

        # 获取历史序列的 semantic_ids: [B, N, K]
        hist_sem_ids, hist_sem_emb = self._get_semantic_id_embeddings(historical_ids)

        # 展平: [B, N, K, D] -> [B, N*K, D]
        N = hist_sem_ids.shape[1]
        K = self.id_length
        D = self.sid_embedding.embedding_dim

        post_sid_emb = hist_sem_emb.view(B, N * K, D)
        post_sid = hist_sem_ids.view(B, N * K)

        # 构建位置编码 (简单的顺序位置编码)
        position_emb = torch.zeros_like(post_sid_emb)
        for i in range(N * K):
            position_emb[:, i, :] = (i + 1) / (N * K)  # 归一化位置

        # 3. 目标序列处理 (训练时)
        if target_semantic_ids is not None:
            # target_semantic_ids: [B, K]，可能包含 -1 表示缺失
            # 需要转换为 embeddings: [B, K, D]

            # 将 -1 替换为 0 (padding_idx)
            target_semantic_ids = torch.where(
                target_semantic_ids == -1,
                torch.zeros_like(target_semantic_ids),
                target_semantic_ids
            )

            target_flat = target_semantic_ids.view(-1)
            target_emb_flat = self.sid_embedding(target_flat)
            target_sid_emb = target_emb_flat.view(B, K, D)

            # 目标序列 padding mask (假设没有 padding)
            target_padding_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
        else:
            target_sid_emb = None
            target_padding_mask = None

        return {
            "user_emb": user_emb,
            "post_sid_emb": post_sid_emb,
            "post_sid": post_sid,
            "target_sid_emb": target_sid_emb,
            "target_padding_mask": target_padding_mask,
            "position_emb": position_emb,
        }

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        """TIGER 训练步骤"""
        # 1. 解析 batch
        seq_features, target_labels = seq_features_from_row(
            batch, self.device, self.gr_output_length
        )
        B = seq_features["historical_ids"].shape[0]

        # 2. 获取 target items 的 Semantic IDs
        target_item_ids = target_labels["target_ids"].squeeze(-1)  # [B]
        target_semantic_ids = self.semantic_id_manager.tokenize(target_item_ids.tolist()).to(self.device)  # [B, K]

        # 将 -1 替换为 0（padding），防止后续索引越界
        target_semantic_ids = torch.where(
            target_semantic_ids == -1,
            torch.zeros_like(target_semantic_ids),
            target_semantic_ids
        )

        # 3. 构建 TIGER 输入
        tiger_inputs = self._build_tiger_inputs(seq_features, target_semantic_ids)

        # 4. 调用 TIGER forward
        # TIGER 返回 logits: [B, K+1, vocab_size] (包含 BOS)
        logits = self.sequence_model(
            user_emb=tiger_inputs["user_emb"],
            post_sid_emb=tiger_inputs["post_sid_emb"],
            post_sid=tiger_inputs["post_sid"],
            target_sid_emb=tiger_inputs["target_sid_emb"],
            target_padding_mask=tiger_inputs["target_padding_mask"],
            position_emb=tiger_inputs["position_emb"],
        )

        # 5. 计算损失
        # logits: [B, K+1, vocab_size]，跳过 BOS
        logits = logits[:, 1:, :]  # [B, K, vocab_size]
        logits_flat = logits.reshape(-1, self.codebook_size)
        targets_flat = target_semantic_ids.reshape(-1)

        # 使用 -1 作为 ignore_index (标记无效的semantic ID)
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=-1,  # 忽略值为-1的目标（无效semantic ID）
        )

        # 6. 记录日志
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # 计算准确率
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == target_semantic_ids).float().mean()
            self.log("train/accuracy", accuracy, on_step=True, on_epoch=True, logger=True)

        return loss

    @torch.inference_mode
    def retrieve(
        self,
        seq_features: dict[str, torch.Tensor],
        top_k: int = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        TIGER 生成式检索 - 多次采样生成 top-k 推荐

        Returns:
            top_k_item_ids: [B, K] 推荐的物品 ID
            top_k_scores: [B, K] 对应的分数
        """
        if top_k is None:
            top_k = self.metrics.topk

        B = seq_features["historical_ids"].shape[0]

        # 1. 构建 TIGER 输入
        tiger_inputs = self._build_tiger_inputs(seq_features, target_semantic_ids=None)

        # 2. 多次采样生成候选
        # 为了生成 top_k 个不同的推荐,需要多次采样
        num_samples = min(top_k * 3, 500)  # 采样 3 倍以保证多样性,最多 500 次
        
        all_item_ids = []
        
        for _ in range(num_samples):
            # 调用 TIGER.generate() 生成 semantic IDs
            generated_semantic_ids, _ = self.sequence_model.generate(
                user_emb=tiger_inputs["user_emb"],
                post_sid_emb=tiger_inputs["post_sid_emb"],
                post_sid=tiger_inputs["post_sid"],
                position_emb=tiger_inputs["position_emb"],
                max_gen_len=self.id_length,
                temperature=temperature,
                top_k=50,
            )
            
            # Decode 每个样本
            batch_item_ids = []
            for b in range(B):
                semantic_id = generated_semantic_ids[b:b+1]  # [1, K]
                item_id = self.semantic_id_manager.decode(semantic_id)[0]
                
                # 过滤无效 ID (None 或 0)
                if item_id is not None and item_id > 0:
                    batch_item_ids.append(item_id)
                else:
                    batch_item_ids.append(0)  # 占位符,后面会被过滤
            
            all_item_ids.append(torch.tensor(batch_item_ids, device=self.device))
        
        # 3. 去重并选择 top-k
        top_k_item_ids = torch.zeros(B, top_k, dtype=torch.long, device=self.device)
        top_k_scores = torch.zeros(B, top_k, device=self.device)
        
        for b in range(B):
            # 收集该样本的所有候选
            candidates = [all_item_ids[i][b].item() for i in range(num_samples)]
            
            # 去重并过滤无效 ID
            unique_items = []
            for item in candidates:
                if item > 0 and item not in unique_items:
                    unique_items.append(item)
            
            # 填充 top-k
            for k in range(min(top_k, len(unique_items))):
                top_k_item_ids[b, k] = unique_items[k]
                top_k_scores[b, k] = 1.0 / (k + 1)  # 简单的分数,按顺序递减
            
            # 如果候选不足 top_k,用随机 item 填充
            if len(unique_items) < top_k:
                # 获取所有有效 item IDs (从 semantic_id_manager)
                all_valid_items = list(self.semantic_id_manager.item_id_to_semantic_ids.keys())
                all_valid_items = [i for i in all_valid_items if i > 0]  # 过滤 0
                
                # 随机选择填充
                remaining = top_k - len(unique_items)
                random_indices = torch.randperm(len(all_valid_items))[:remaining]
                for idx, k in enumerate(range(len(unique_items), top_k)):
                    random_item = all_valid_items[random_indices[idx]]
                    if random_item not in unique_items:
                        top_k_item_ids[b, k] = random_item
                        top_k_scores[b, k] = 1.0 / (k + 1)

        return top_k_item_ids, top_k_scores

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """验证步骤"""
        seq_features, target_labels = seq_features_from_row(
            batch, self.device, self.gr_output_length
        )

        # 生成式检索
        top_k_ids, _top_k_scores = self.retrieve(
            seq_features,
            top_k=self.metrics.topk,
        )

        # 日志（仅第一个 batch）
        if batch_idx == 0:
            target_item_ids = target_labels["target_ids"]
            target_semantic_ids = self.semantic_id_manager.tokenize(
                target_item_ids.squeeze(-1).tolist()
            )

            logger.info(f"\n[Validation Debug] Batch {batch_idx}")
            for i in range(min(3, len(target_item_ids))):
                tgt_item = target_item_ids[i].item()
                tgt_sem = target_semantic_ids[i].tolist()
                pred_item = top_k_ids[i, 0].item()

                # 尝试获取预测项的 semantic ID
                try:
                    pred_sem = self.semantic_id_manager.tokenize([pred_item])[0].tolist()
                except:
                    pred_sem = "Unknown"

                logger.info(f"  Sample {i}:")
                logger.info(f"    Target: Item {tgt_item} -> {tgt_sem}")
                logger.info(f"    Pred  : Item {pred_item} -> {pred_sem}")

        # 更新指标
        self.metrics.update(top_k_ids=top_k_ids, target_ids=target_labels["target_ids"])

    def on_validation_epoch_end(self) -> None:
        """验证结束"""
        results = self.metrics.compute()
        for k, v in results.items():
            self.log(f"val/{k}", v, on_epoch=True, prog_bar=True, logger=True)
        logger.info(f"val results: {results}")
        self.metrics.reset()

        if self.lr_monitor_metric and "/" in self.lr_monitor_metric:
            metric_name = self.lr_monitor_metric.split("/")[1]
            if metric_name in results:
                return results[metric_name]

    def on_test_epoch_start(self):
        self.metrics.reset()

    def on_test_epoch_end(self):
        results = self.metrics.compute()
        for k, v in results.items():
            self.log(f"test/{k}", v, on_epoch=True, prog_bar=True, logger=True)
        logger.info(f"test results: {results}")
        self.metrics.reset()

    def test_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        self.validation_step(batch, batch_idx)

    def predict_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        self.validation_step(batch, batch_idx)

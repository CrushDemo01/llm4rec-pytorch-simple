import torch

from llm4rec_pytorch_simple.models.generative_recommenders import GenerativeRecommenders
from llm4rec_pytorch_simple.utils.features import get_current_embeddings_simple, seq_features_from_row
from llm4rec_pytorch_simple.utils.pylogger import RankedLogger

logger = RankedLogger(__name__)


class RetrievalModule(GenerativeRecommenders):
    @torch.inference_mode  # PyTorch中的一个装饰器，用于优化推理阶段的性能。
    def retrieve(
        self,
        seq_features: dict[str, torch.Tensor],
        filter_past_ids: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the top-k items for the given sequence features.
        """
        seq_embeddings = self.forward(
            past_ids=seq_features["historical_ids"], past_embeddings=seq_features["historical_id_embeddings"]
        )  # [B, X]

        # 获取当前序列的最后一个有效位置的嵌入
        current_embeddings = get_current_embeddings_simple(seq_features["historical_lengths"], seq_embeddings)

        top_k_ids, top_k_scores = self.embedding.get_top_k_outputs(
            query_embeddings=current_embeddings,
            k=self.metrics.topk,
            invalid_ids=(seq_features["historical_ids"] if filter_past_ids else None),
        )

        return top_k_ids, top_k_scores

    def on_validation_epoch_start(self) -> None:
        # 每次fit之前都会先进行val，而不是train
        self.metrics.reset()  # 重置指标

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        # 做一些序列上的处理
        # self.gr_output_length说明最少一个
        seq_features, target_labels = seq_features_from_row(batch, self.device, self.gr_output_length)

        # 获取embedding
        historical_id_embeddings = self.embedding.get_item_embeddings(seq_features["historical_ids"])
        seq_features["historical_id_embeddings"] = historical_id_embeddings

        # forward pass
        top_k_ids, _top_k_scores = self.retrieve(seq_features)
        self.metrics.update(top_k_ids=top_k_ids, target_ids=target_labels["target_ids"])

    def on_validation_epoch_end(self) -> None:
        # 在验证集上评估模型，计算各种指标（准确率、召回率等）
        results = self.metrics.compute()
        for k, v in results.items():
            self.log(
                f"val/{k}",  # 指标名：val/recall@10, val/ndcg@10, val/mrr
                v,  # 指标值：0.234, 0.156, 0.189
                on_epoch=True,  # ✅ epoch结束时记录（适合验证指标）
                prog_bar=True,  # ✅ 显示在进度条
                logger=True,  # ✅ 发送到CSV和TensorBoard
            )
        logger.info(f"val results: {results}")
        self.metrics.reset()

        # 返回监控指标给学习率调度器
        # 从 "val/metric_name" 格式中提取 metric_name
        if self.lr_monitor_metric and "/" in self.lr_monitor_metric:
            metric_name = self.lr_monitor_metric.split("/")[1]
            if metric_name in results:
                return results[metric_name]

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        # 做一些序列上的处理
        seq_features, target_labels = seq_features_from_row(batch, self.device, self.gr_output_length)

        # 在历史行为id后面添加上target_ids，add target_ids at the end of the past_ids
        seq_features["historical_ids"].scatter_(
            dim=1,
            index=seq_features["historical_lengths"].view(-1, 1),
            src=target_labels["target_ids"].view(-1, 1),
        )

        # 获取embedding
        historical_id_embeddings = self.embedding.get_item_embeddings(seq_features["historical_ids"])
        seq_features["historical_id_embeddings"] = historical_id_embeddings

        # forward pass
        seq_embeddings = self.forward(
            past_ids=seq_features["historical_ids"], past_embeddings=seq_features["historical_id_embeddings"]
        )  # [B, X]
        # ============ 简化的损失计算 ============
        _B, _N, _D = seq_embeddings.shape

        # 5. 错位对齐（密集格式）
        """
        序列: [item1, item2, item3, item4]
        索引:    0      1      2      3

        模型输出 (output_emb):
        位置0 → emb0  预测→ item2
        位置1 → emb1  预测→ item3
        位置2 → emb2  预测→ item4
        位置3 → emb3  ❌ 没有目标,丢弃

        目标 (target_emb):
        item2的嵌入
        item3的嵌入
        item4的嵌入

        对齐:
        emb0 vs item2_emb
        emb1 vs item3_emb
        emb2 vs item4_emb
        """
        output_emb = seq_embeddings[:, :-1, :]  # [B, N-1, D] 输出
        target_ids_shifted = seq_features["historical_ids"][:, 1:]  # [B, N-1] 目标ID
        target_emb = historical_id_embeddings[:, 1:, :]  # [B, N-1, D] 目标嵌入
        # 6. 创建掩码（过滤填充位置）
        mask = (target_ids_shifted != 0).float()  # [B, N-1]
        loss = self.loss(
            negatives_sampler=self.negative_sampler,
            output_embeddings=output_emb,
            pos_embeddings=target_emb,
            positive_ids=target_ids_shifted,  # 正样本id，形状为 (batch_size,)，用于采样负样本
            supervision_mask=mask,  # mask，用于过滤掉填充位置
            num_to_sample=4,  # 每个正样本采样的负样本数量
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_test_epoch_start(self):
        self.metrics.reset()

    def on_test_epoch_end(self):
        results = self.metrics.compute()
        for k, v in results.items():
            self.log(
                f"test/{k}",  # 指标名：val/recall@10, val/ndcg@10, val/mrr
                v,  # 指标值：0.234, 0.156, 0.189
                on_epoch=True,  # ✅ epoch结束时记录（适合验证指标）
                prog_bar=True,  # ✅ 显示在进度条
                logger=True,  # ✅ 发送到CSV和TensorBoard
            )
            logger.info(f"val results: {results}")
            self.metrics.reset()

    def test_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        # 在测试集上评估模型，计算各种指标（准确率、召回率等）
        self.validation_step(batch, batch_idx)

    def predict_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        # 在新数据上生成预测，不需要标签，不计算指标
        self.validation_step(batch, batch_idx)

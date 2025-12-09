from typing import Any

import hydra
import lightning as L
import torch
import torchmetrics
from omegaconf import DictConfig
from llm4rec_pytorch_simple.data.rec_dataset import RecoDataModule
from llm4rec_pytorch_simple.models.embeddings.embeddings import EmbeddingModule
from llm4rec_pytorch_simple.utils.pylogger import RankedLogger

logger = RankedLogger(__name__)


class GenerativeRecommenders(L.LightningModule):
    """
    生成式推荐模型基类
    - 数据模块：RecoDataModule
    - embedding模块：LocalEmbedding
    - 负采样模块：negative_samples
    - 模型：archs
    - 损失函数：losses
    - 指标：metrics
    """

    def __init__(
        self,
        embedding: DictConfig,
        negative_sampler: DictConfig,
        sequence_model: DictConfig,
        loss: DictConfig,
        metrics: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        positional_emb: DictConfig,
        gr_output_length: int,  # 生成式推荐的输出序列长度
        item_embedding_dim: int,  # 项目嵌入的维度
        datamodule: DictConfig = None,
        norm_type: str = "layer",  # 归一化类型
        torch_compile: bool = True,  # 是否编译模型以提升性能
        **kwargs,  # 接收并忽略配置文件中的下划线前缀参数 (如 _num_items, _max_sequence_len 等)
    ):
        super().__init__()
        # 初始化优化器，支持配置对象或实例
        # 初始化优化器，支持配置对象或实例
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler

        # 从 scheduler 配置中提取 monitor 指标(如果存在)
        # 用于学习率调度器监控验证指标
        self.lr_monitor_metric = getattr(scheduler, "monitor", "val/hr@10")

        # 保存生成式推荐的关键参数
        assert gr_output_length >= 1, "gr_output_length must be >= 1"
        self.gr_output_length: int = gr_output_length
        self.item_embedding_dim: int = item_embedding_dim
        self.torch_compile: bool = torch_compile
        self.norm_type: str = norm_type

        self.__hydra_init_submodules(
            datamodule,
            embedding,
            negative_sampler,
            sequence_model,
            loss,
            metrics,
            positional_emb,
        )

    def __hydra_init_submodules(
        self,
        datamodule: DictConfig,
        embedding: DictConfig,
        negative_sampler: DictConfig,
        sequence_model: DictConfig,
        loss: DictConfig,
        metrics: DictConfig,
        positional_emb: DictConfig,
    ) -> None:
        if datamodule is not None:
            self.datamodule: RecoDataModule = hydra.utils.instantiate(datamodule)
        self.embedding: EmbeddingModule = hydra.utils.instantiate(embedding)
        self.negative_sampler: torch.nn.Module = hydra.utils.instantiate(negative_sampler)
        # 设置 negative_sampler 的 embedding 层
        self.negative_sampler._item_emb = self.embedding.embeddings
        self.sequence_model: torch.nn.Module = hydra.utils.instantiate(sequence_model)
        self.loss: torch.nn.Module = hydra.utils.instantiate(loss)
        self.metrics: torchmetrics.Metric = hydra.utils.instantiate(metrics)
        self.positional_emb: torch.nn.Module = hydra.utils.instantiate(positional_emb)

    def setup(self, stage: str) -> None:
        # 设置数据集，根据训练、验证或测试阶段加载数据，在fit, test, predict 等都会自动调用
        if self.torch_compile and stage == "fit":  # 这里在fit才编译
            self.embedding = torch.compile(self.embedding)
            self.negative_sampler = torch.compile(self.negative_sampler)
            self.sequence_model = torch.compile(self.sequence_model)
            self.loss = torch.compile(self.loss)
            self.positional_emb = torch.compile(self.positional_emb)

    def configure_optimizers(self) -> dict[str, Any]:
        """
        配置优化器和学习率调度器
        1. 优化器需要模型参数 (self.parameters()),只能在模型初始化后获取
        2. 调度器需要优化器实例,只能在优化器创建后获取
        3. 使用 _partial_: true + 两步实例化,可以灵活地在正确的时机传入这些依赖参数
        如果不这样改,训练脚本会在 configure_optimizers()阶段报错,提示缺少必需参数。
        """
        # 实例化优化器(使用 _partial_: true,返回部分函数,需要传入 params)
        optimizer_partial = hydra.utils.instantiate(self.optimizer_cfg)
        optimizer = optimizer_partial(params=self.parameters())

        # 实例化调度器(使用 _partial_: true,返回部分函数,需要传入 optimizer)
        scheduler_partial = hydra.utils.instantiate(self.scheduler_cfg)
        scheduler = scheduler_partial(optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.lr_monitor_metric,  # 使用配置的监控指标
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        获取模型状态字典，优化存储空间

        重写PyTorch的state_dict方法，排除某些不需要保存的模块参数，以减少模型文件大小。
        这种优化特别适用于那些可以在加载时重新创建的模块，如损失函数、评估指标等。

        排除的模块：
        - similarity: 相似度计算模块
        - negatives_sampler: 负采样器
        - candidate_index: 候选项目索引
        - loss: 损失函数
        - metrics: 评估指标

        参数:
            destination: 目标字典(可选)
            prefix: 参数名前缀
            keep_vars: 是否保留变量属性(梯度)

        返回:
            优化后的状态字典，排除了指定模块的参数
        """
        # 调用父类方法获取完整的状态字典
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        # 定义不需要保存的模块列表
        modules_to_exclude = [
            "similarity",
            "negatives_sampler",
            "candidate_index",
            "loss",
            "metrics",
        ]

        # 遍历状态字典，删除不需要保存的模块参数
        keys_to_remove = []
        for module_name in modules_to_exclude:
            for key in state_dict:
                if key.startswith(prefix + module_name):
                    keys_to_remove.append(key)

        # 从状态字典中删除不需要保存的模块参数
        for key in keys_to_remove:
            del state_dict[key]

        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        """
        加载模型状态字典

        重写PyTorch的load_state_dict方法，设置strict=False以兼容优化后的状态字典。
        由于state_dict方法排除了某些模块的参数，加载时需要允许不匹配的键。

        参数:
            state_dict: 要加载的状态字典
            strict: 是否严格匹配所有键，设为False以忽略缺失的键
        """
        # 由于我们在state_dict中删除了一些键，加载时需要设置strict=False
        super().load_state_dict(state_dict, strict=False)

    def _normalize(self, embeddings: torch.Tensor, norm_type: str = "layer") -> torch.Tensor:
        """
        对嵌入向量进行归一化
        """
        if norm_type == "layer":
            return torch.nn.functional.layer_norm(embeddings, (self.item_embedding_dim,))
        elif norm_type == "l2":
            return torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        else:
            raise ValueError(f"Invalid norm_type: {norm_type}")

    def forward(self, past_ids: int, past_embeddings: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        user_embeddings, _valid_mask = self.positional_emb(past_ids, past_embeddings)
        output_embeddings = self.sequence_model(user_embeddings, past_ids)
        output_embeddings = self._normalize(output_embeddings, self.norm_type)
        return output_embeddings


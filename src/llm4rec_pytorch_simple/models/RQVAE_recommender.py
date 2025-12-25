import torch
import torch.nn.functional as F
from omegaconf import DictConfig
import hydra
from llm4rec_pytorch_simple.utils.pylogger import RankedLogger
import lightning as L
import torchmetrics

logger = RankedLogger(__name__)

class RQVAERecommender(L.LightningModule):
    def __init__(
        self,
        negative_sampler: DictConfig,
        sequence_model: DictConfig,
        loss: DictConfig,
        metrics: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        gr_output_length: int,
        item_embedding_dim: int,
        norm_type: str = "layer",
        torch_compile: bool = True,
        **submodules,
    ):
        """
        必须的项都单独列出来；一些基于 nn.Module 的项，放在 **submodules 中
        """
        super().__init__()

        # 初始化优化器，支持配置对象或实例
        self.optimizer_cfg: DictConfig = optimizer
        self.scheduler_cfg: DictConfig = scheduler
        # 用于学习率调度器监控验证指标
        self.lr_monitor_metric: str = getattr(scheduler, "monitor", "val/hr@10")
        self.torch_compile: bool = torch_compile
        self.gr_output_length: int = gr_output_length
        self.item_embedding_dim: int = item_embedding_dim
        self.norm_type: str = norm_type
        self.submodules: DictConfig = submodules
        
        self.__hydra_init_submodules(
            negative_sampler,
            sequence_model,
            loss,
            metrics,
            submodules
        )

    def __hydra_init_submodules(
        self,
        negative_sampler: DictConfig,
        sequence_model: DictConfig,
        loss: DictConfig,
        metrics: DictConfig,
        submodules
    ) -> None:
        self.negative_sampler: torch.nn.Module = hydra.utils.instantiate(negative_sampler)
        self.sequence_model: torch.nn.Module = hydra.utils.instantiate(sequence_model)
        self.loss: torch.nn.Module = hydra.utils.instantiate(loss)
        self.metrics: torchmetrics.Metric = hydra.utils.instantiate(metrics)

        for k, v in submodules.items():
            if isinstance(v, (DictConfig, dict)):
                # 只实例化带有 _target_ 的配置
                if "_target_" in v:
                    setattr(self, k, hydra.utils.instantiate(v))
                else:
                    logger.warning(f"Submodule '{k}' 没有 _target_，将作为配置字典直接设置")
                    setattr(self, k, v)
            else:
                # 已经是实例化对象，直接设置
                setattr(self, k, v)

    def setup(self, stage: str) -> None:
        # 设置数据集，根据训练、验证或测试阶段加载数据，在fit, test, predict 等都会自动调用
        # 只编译一个模型
        if self.torch_compile:  # 这里在fit才编译
            self.sequence_model = torch.compile(self.sequence_model)

    def configure_optimizers(self):
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
        优化存储：排除不必要的模块（loss/metrics/sampler 等可重建模块）
        """
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        # 定义不保存的模块前缀
        exclude_prefixes = ("similarity", "negative_sampler", "candidate_index", "loss", "metrics")
        
        # 一次性过滤：使用集合推导式找到所有需要排除的键
        keys_to_remove = {
            key for key in state_dict.keys()
            if any(key.startswith(prefix + mod) for mod in exclude_prefixes)
        }
        
        # 批量删除
        for key in keys_to_remove:
            del state_dict[key]
        
        if keys_to_remove:
            logger.debug(f"已排除 {len(keys_to_remove)} 个不必要的参数")

        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        """
        加载状态字典，兼容优化后的 checkpoint（缺失 loss/metrics 等模块）
        """
        # 由于 state_dict() 排除了部分模块，这里强制 strict=False
        result = super().load_state_dict(state_dict, strict=False)
        
        # 记录缺失和多余的键（用于调试）
        if result.missing_keys:
            logger.debug(f"加载时缺失 {len(result.missing_keys)} 个键（正常）")
        if result.unexpected_keys:
            logger.warning(f"发现 {len(result.unexpected_keys)} 个多余的键: {result.unexpected_keys[:5]}")
        
        return result

    def _normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """对嵌入向量归一化（使用初始化时指定的 norm_type）"""
        if self.norm_type == "layer":
            return F.layer_norm(embeddings, (self.item_embedding_dim,))
        elif self.norm_type == "l2":
            return F.normalize(embeddings, p=2, dim=-1)
        else:
            raise ValueError(f"不支持的归一化类型: {self.norm_type}")

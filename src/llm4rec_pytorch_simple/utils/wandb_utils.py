"""WandB 辅助工具模块

提供 WandB 实验追踪的辅助功能，包括：
- 配置参数表格记录
- 指标汇总记录
- 实验对比辅助
"""

from typing import Any, Dict, Optional

import wandb
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf


def log_config_table(logger: WandbLogger, config: DictConfig) -> None:
    """记录配置参数表格到 WandB
    
    将 Hydra 配置转换为表格形式，方便在 WandB 网页端查看和对比。
    
    Args:
        logger: WandB logger 实例
        config: Hydra 配置对象
    """
    if not isinstance(logger, WandbLogger):
        return
    
    # 将 DictConfig 转换为普通字典
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # 展平嵌套配置为单层字典
    flat_config = _flatten_dict(config_dict)
    
    # 创建配置表格
    config_table = wandb.Table(
        columns=["Parameter", "Value"],
        data=[[k, str(v)] for k, v in sorted(flat_config.items())]
    )
    
    # 记录到 WandB
    logger.experiment.log({"config_table": config_table})


def log_metrics_summary(
    logger: WandbLogger,
    metrics: Dict[str, Any],
    prefix: str = "final"
) -> None:
    """记录最终指标汇总
    
    Args:
        logger: WandB logger 实例
        metrics: 指标字典
        prefix: 指标前缀
    """
    if not isinstance(logger, WandbLogger):
        return
    
    summary_metrics = {
        f"{prefix}/{k}": v for k, v in metrics.items()
        if isinstance(v, (int, float))
    }
    
    logger.experiment.log(summary_metrics)


def create_comparison_table(
    run_ids: list[str],
    project: str,
    entity: Optional[str] = None
) -> wandb.Table:
    """创建实验对比表格（可选功能）
    
    从 WandB API 获取多个实验的配置和指标，生成对比表格。
    
    Args:
        run_ids: 实验 ID 列表
        project: WandB 项目名称
        entity: WandB 团队名称（可选）
    
    Returns:
        WandB 表格对象
    """
    api = wandb.Api()
    
    runs_data = []
    for run_id in run_ids:
        run_path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
        run = api.run(run_path)
        
        runs_data.append({
            "run_id": run_id,
            "name": run.name,
            "config": run.config,
            "summary": run.summary._json_dict
        })
    
    # 提取所有配置键和指标键
    all_config_keys = set()
    all_metric_keys = set()
    
    for data in runs_data:
        all_config_keys.update(data["config"].keys())
        all_metric_keys.update(data["summary"].keys())
    
    # 构建表格列
    columns = ["run_id", "name"] + sorted(all_config_keys) + sorted(all_metric_keys)
    
    # 构建表格数据
    table_data = []
    for data in runs_data:
        row = [data["run_id"], data["name"]]
        row.extend([data["config"].get(k, "N/A") for k in sorted(all_config_keys)])
        row.extend([data["summary"].get(k, "N/A") for k in sorted(all_metric_keys)])
        table_data.append(row)
    
    return wandb.Table(columns=columns, data=table_data)


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """展平嵌套字典
    
    Args:
        d: 嵌套字典
        parent_key: 父键名
        sep: 分隔符
    
    Returns:
        展平后的字典
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # 列表转换为字符串
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    
    return dict(items)

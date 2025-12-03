from collections.abc import Sequence
from pathlib import Path
from lightning.pytorch.callbacks import RichProgressBar
import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt
from rich.table import Table

from llm4rec_pytorch_simple.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    :param cfg: A DictConfig composed by Hydra.
    :param print_order: Determines in what order config components are printed. Default is ``("data", "model",
    "callbacks", "logger", "trainer", "paths", "extras")``.
    :param resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
    :param save_to_file: Whether to export config to the hydra output folder. Default is ``False``.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config.

    :param cfg: A DictConfig composed by Hydra.
    :param save_to_file: Whether to export tags to the hydra output folder. Default is ``False``.
    """
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            error_msg = "Specify tags before launching a multirun!"
            raise ValueError(error_msg)

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)


class CustomRichProgressBar(RichProgressBar):
    """自定义 RichProgressBar，以表格形式显示验证指标。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_metrics = {}

    def get_metrics(self, trainer, pl_module):
        # 获取所有指标
        items = super().get_metrics(trainer, pl_module)
        # 在验证阶段结束时，以表格形式打印指标
        if "val/hr@10" in items and trainer.state.stage == "validate":
            self.val_metrics.update(items)
            # 清除进度条中的验证指标，避免重复显示
            for key in list(items):
                if key.startswith("val/"):
                    items.pop(key)
        return items

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        if not self.val_metrics:
            return

        # 创建表格
        table = Table(show_header=True, header_style="bold magenta")
        
        # 收集数据
        row_data = []
        headers = []
        
        for k, v in sorted(self.val_metrics.items()):
            if k.startswith("val/"):
                headers.append(k)
                # 尝试将Tensor转换为浮点数
                if hasattr(v, "item"):
                    v = v.item()
                row_data.append(f"{v:.4f}")
        
        # 添加列
        for header in headers:
            table.add_column(header, justify="center")
            
        # 添加数据行
        if row_data:
            table.add_row(*row_data)

        # 打印表格
        # RichProgressBar 内部使用 self._console 或 self.progress.console
        # 但为了安全起见，我们可以直接使用 rich.print 或者 trainer.console (如果存在)
        # 这里我们使用 rich.print，因为它最通用
        rich.print(table)
        
        # 清空缓存的指标
        self.val_metrics.clear()

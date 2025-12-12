from typing import Any

from lightning.pytorch.loggers import WandbLogger
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from llm4rec_pytorch_simple.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters
        - WandB config table (if WandB logger is used)

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    cfg_container = OmegaConf.to_container(cfg)
    hparams["model"] = cfg_container["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    hparams["data"] = cfg_container["data"]
    hparams["trainer"] = cfg_container["trainer"]

    hparams["callbacks"] = cfg_container.get("callbacks")
    hparams["extras"] = cfg_container.get("extras")

    hparams["task_name"] = cfg_container.get("task_name")
    hparams["tags"] = cfg_container.get("tags")
    hparams["ckpt_path"] = cfg_container.get("ckpt_path")
    hparams["seed"] = cfg_container.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
        
        # 为 WandB logger 额外记录配置表格，方便对比实验
        if isinstance(logger, WandbLogger):
            try:
                from llm4rec_pytorch_simple.utils.wandb_utils import log_config_table
                log_config_table(logger, cfg)
                log.info("WandB config table logged successfully")
            except Exception as e:
                log.warning(f"Failed to log WandB config table: {e}")

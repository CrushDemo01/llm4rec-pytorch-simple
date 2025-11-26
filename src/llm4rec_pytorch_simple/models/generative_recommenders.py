import lightning as L
import hydra
from omegaconf import DictConfig
from llm4rec_pytorch_simple.utils.pylogger import RankedLogger
from llm4rec_pytorch_simple.data.rec_dataset import RecoDataModule
from llm4rec_pytorch_simple.models.embeddings.embeddings import LocalEmbedding
from llm4rec_pytorch_simple.models.negative_samples.negative_samples import LocalNegativeSamples
from llm4rec_pytorch_simple.models.archs.SASRec import SASRec
from llm4rec_pytorch_simple.models.losses.losses import BCELoss
from llm4rec_pytorch_simple.models.metrics.metrics import RankingMetrics, RetrievalMetrics


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

    def __init__(self,
        datamodule: DictConfig,
        embeddings: DictConfig,
        negative_sampler: DictConfig,
        sequence_model: DictConfig,
        loss: DictConfig,
        metrics: DictConfig,
                 ):
        super().__init__()
        pass

    def __hydra_init_submodules(self, 
        datamodule: DictConfig,
        embeddings: DictConfig,
        negative_sampler: DictConfig,
        sequence_model: DictConfig,
        loss: DictConfig,
        metrics: DictConfig):
        # 初始化各个子模块
        pass



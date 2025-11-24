import torch

from llm4rec_pytorch_simple.models.metrics.metrics import RankingMetrics, RetrievalMetrics
from llm4rec_pytorch_simple.utils.pylogger import RankedLogger

logger = RankedLogger(__name__)


class RetrievalModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ranking_metrics = RankingMetrics(config.at_k_list)
        self.retrieval_metrics = RetrievalMetrics(config.at_k_list)

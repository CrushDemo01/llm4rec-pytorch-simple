import torch

from llm4rec_pytorch_simple.utils.pylogger import RankedLogger

logger = RankedLogger(__name__)


class LocalEmbedding(torch.nn.Module):
    """Local embedding layer.

    :param num_embeddings: The number of embeddings.
    :param embedding_dim: The dimension of the embeddings.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = 0) -> None:
        """Initialize a `LocalEmbedding` module.

        :param num_embeddings: The number of embeddings.
        :param embedding_dim: The dimension of the embeddings.
        :param padding_idx: The index to use for padding. Defaults to 0.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(num_embeddings + 1, embedding_dim, padding_idx=padding_idx)
        self.__init_params()

    def __init_params(self) -> None:
        """Initialize the parameters of the module."""
        torch.nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of embeddings.
        """
        return self.embedding(x)

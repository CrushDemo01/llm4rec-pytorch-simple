import abc
from typing import Optional

import torch

from llm4rec_pytorch_simple.models.embeddings.topk import MIPSBruteForceTopK
from llm4rec_pytorch_simple.utils.pylogger import RankedLogger

logger = RankedLogger(__name__)


class EmbeddingModule(torch.nn.Module):
    def __init__(
        self,
        ids: torch.Tensor,  # 候选物品ID列表，形状为(X,)
        top_k_module: MIPSBruteForceTopK,
    ):
        super().__init__()
        # 注册ID为buffer，不参与训练但会随模型保存
        # unsqueeze(0) 增加batch维度，变为(1, X)
        self.register_buffer("_ids", torch.as_tensor(ids).unsqueeze(0))
        self.top_k_module = top_k_module
        self._embeddings_t = None  # 子类需要设置

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    def get_top_k_outputs(
        self,
        query_embeddings: torch.Tensor,
        k: Optional[int] = None,
        invalid_ids: Optional[torch.Tensor] = None,
    ):
        """
        根据查询嵌入向量，找到最相似的K个候选物品，同时过滤掉无效ID
        - 无效ID是用户历史点击的item_ids
        """
        # 计算需要过滤的ID数量
        max_num_invalid_ids = 0
        if invalid_ids is not None:
            max_num_invalid_ids = invalid_ids.size(1)

        # 计算实际需要检索的数量 = k + 需要过滤的数量
        # 这样可以确保过滤后仍有k个有效结果
        k_prime = k + max_num_invalid_ids

        # 执行topk搜索
        top_k_prime_scores, top_k_prime_ids = self.top_k_module(
            query_embeddings=query_embeddings,
            item_embeddings_t=self._embeddings_t,  # (D, X) 转置后的嵌入
            item_ids=self._ids,  # (1, X) 或 (B, X) 物品ID
            k=k_prime,  # 检索k_prime个结果
            sorted=True,  # 按分数排序
        )

        # 过滤无效物品（核心逻辑）
        if invalid_ids is not None:
            top_k_ids, top_k_scores = _filter_invalid_ids_simple(
                top_k_prime_ids=top_k_prime_ids,
                top_k_prime_scores=top_k_prime_scores,
                invalid_ids=invalid_ids,
                k=k,
            )
        else:
            top_k_scores = top_k_prime_scores[:, :k]
            top_k_ids = top_k_prime_ids[:, :k]

        return top_k_ids, top_k_scores


def _filter_invalid_ids_simple(
    top_k_prime_ids: torch.Tensor,  # (B, k_prime) 初步检索的ID
    top_k_prime_scores: torch.Tensor,  # (B, k_prime) 对应的分数
    invalid_ids: torch.Tensor,  # (B, N_0) 需要过滤的ID
    k: int,  # 最终需要的数量
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    简化版：过滤掉无效ID，返回Top-K有效结果

    例子：
    top_k_prime_ids = [[5, 2, 8, 2, 1], [3, 7, 1, 9, 4]]  # 初步检索5个
    invalid_ids = [[2, 8], [1, 3]]                        # 用户已看过的
    k = 3                                                 # 最终要3个

    结果：
    top_k_ids = [[5, 1, 0], [7, 9, 4]]  # 过滤后的Top-3（0是填充）
    """
    B, k_prime = top_k_prime_ids.shape

    # 循环版本（最直观，适合理解）
    result_ids = []
    result_scores = []

    for i in range(B):  # 对每个用户
        valid_ids = []
        valid_scores = []

        # 遍历初步检索结果
        for j in range(k_prime):
            item_id = top_k_prime_ids[i, j].item()
            item_score = top_k_prime_scores[i, j].item()

            # 检查是否在无效列表中
            if item_id not in invalid_ids[i]:
                valid_ids.append(item_id)
                valid_scores.append(item_score)

                # 已经找到k个，停止
                if len(valid_ids) == k:
                    break

        # 如果不足k个，用0填充
        while len(valid_ids) < k:
            valid_ids.append(0)
            valid_scores.append(0.0)

        result_ids.append(valid_ids)
        result_scores.append(valid_scores)

    return (
        torch.tensor(result_ids, dtype=top_k_prime_ids.dtype, device=top_k_prime_ids.device),
        torch.tensor(result_scores, dtype=top_k_prime_scores.dtype, device=top_k_prime_scores.device),
    )


class LocalEmbedding(EmbeddingModule):
    """Local embedding layer.

    :param num_items: The number of embeddings.
    :param embedding_dim: The dimension of the embeddings.
    """

    def __init__(
        self, num_items: int, embedding_dim: int, padding_idx: int = 0, top_k_module: MIPSBruteForceTopK = None
    ) -> None:
        """Initialize a `LocalEmbedding` module.

        :param num_items: The number of embeddings.
        :param embedding_dim: The dimension of the embeddings.
        :param padding_idx: The index to use for padding. Defaults to 0.
        :param top_k_module: TopK module for retrieval.
        """
        # 保存参数供后续使用
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # 准备父类初始化参数
        ids = torch.arange(1, num_items + 1)
        if top_k_module is None:
            top_k_module = MIPSBruteForceTopK()

        # 调用父类EmbeddingModule的__init__
        super().__init__(ids=ids, top_k_module=top_k_module)

        # 创建embedding层
        self.embeddings = torch.nn.Embedding(num_items + 1, embedding_dim, padding_idx=padding_idx)
        self.__init_params()

        # 设置转置的embeddings用于检索(只包含有效物品,排除padding_idx=0)
        # embeddings.weight shape: (num_items+1, embedding_dim)
        # 取索引1到num_items的embeddings,然后转置
        # 使用 self.embeddings.weight.data[1:] 排除第一个 padding embedding
        self._embeddings_t = self.embeddings.weight.data[1:].t()  # (embedding_dim, num_items)

    def __init_params(self) -> None:
        """Initialize the parameters of the module."""
        torch.nn.init.normal_(self.embeddings.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of embeddings.
        """
        return self.embeddings(x)

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings(item_ids)


# ==================== 测试函数 ====================
def test_local_embedding():
    """测试LocalEmbedding的基本功能"""
    print("\n" + "=" * 50)
    print("测试1: LocalEmbedding基本功能")
    print("=" * 50)

    num_items = 100
    embedding_dim = 64

    # 创建embedding模块
    embedding = LocalEmbedding(num_items=num_items, embedding_dim=embedding_dim)

    # 测试forward
    item_ids = torch.tensor([[1, 2, 3, 0], [5, 10, 0, 0]])  # (2, 4)
    embeddings = embedding(item_ids)

    print(f"✓ 输入item_ids shape: {item_ids.shape}")
    print(f"✓ 输出embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (2, 4, embedding_dim), f"期望shape (2, 4, {embedding_dim}), 得到 {embeddings.shape}"
    print("✓ Embedding forward测试通过")

    # 测试get_item_embeddings
    single_item = torch.tensor([5])
    single_emb = embedding.get_item_embeddings(single_item)
    print(f"✓ 单个物品embedding shape: {single_emb.shape}")
    assert single_emb.shape == (1, embedding_dim)
    print("✓ get_item_embeddings测试通过\n")


def test_filter_invalid_ids():
    """测试_filter_invalid_ids_simple函数"""
    print("=" * 50)
    print("测试2: 过滤无效ID功能")
    print("=" * 50)

    # 测试数据
    top_k_prime_ids = torch.tensor([[5, 2, 8, 2, 1], [3, 7, 1, 9, 4]])
    top_k_prime_scores = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5], [0.95, 0.85, 0.75, 0.65, 0.55]])
    invalid_ids = torch.tensor([[2, 8], [1, 3]])
    k = 3

    print(f"初步检索结果: {top_k_prime_ids.tolist()}")
    print(f"无效ID列表: {invalid_ids.tolist()}")
    print(f"需要返回Top-{k}")

    # 执行过滤
    result_ids, result_scores = _filter_invalid_ids_simple(
        top_k_prime_ids=top_k_prime_ids, top_k_prime_scores=top_k_prime_scores, invalid_ids=invalid_ids, k=k
    )

    print(f"\n过滤后的ID: {result_ids.tolist()}")
    print(f"过滤后的分数: {result_scores.tolist()}")

    # 验证第一个batch
    expected_ids_0 = [5, 1]  # 2和8被过滤
    assert result_ids[0].tolist()[:2] == expected_ids_0, f"期望 {expected_ids_0}, 得到 {result_ids[0].tolist()}"

    # 验证第二个batch
    expected_ids_1 = [7, 9, 4]  # 1和3被过滤
    assert result_ids[1].tolist() == expected_ids_1, f"期望 {expected_ids_1}, 得到 {result_ids[1].tolist()}"

    print("✓ 过滤功能测试通过\n")


def test_top_k_retrieval():
    """测试Top-K检索功能(不带无效ID过滤)"""
    print("=" * 50)
    print("测试3: Top-K检索(无过滤)")
    print("=" * 50)

    num_items = 50
    embedding_dim = 32
    batch_size = 2
    k = 5

    # 创建embedding模块
    embedding = LocalEmbedding(num_items=num_items, embedding_dim=embedding_dim)

    # 创建查询向量
    query_embeddings = torch.randn(batch_size, embedding_dim)

    print(f"查询向量 shape: {query_embeddings.shape}")
    print(f"候选物品数量: {num_items}")
    print(f"检索Top-{k}")

    # 执行检索
    top_k_ids, top_k_scores = embedding.get_top_k_outputs(query_embeddings=query_embeddings, k=k, invalid_ids=None)

    print(f"\n✓ Top-K IDs shape: {top_k_ids.shape}")
    print(f"✓ Top-K scores shape: {top_k_scores.shape}")
    print(f"✓ Top-K IDs 示例: {top_k_ids[0].tolist()}")
    print(f"✓ Top-K scores 示例: {top_k_scores[0].tolist()[:3]}...")

    assert top_k_ids.shape == (batch_size, k), f"期望shape ({batch_size}, {k}), 得到 {top_k_ids.shape}"
    assert top_k_scores.shape == (batch_size, k), f"期望shape ({batch_size}, {k}), 得到 {top_k_scores.shape}"

    # 验证分数是降序排列
    for i in range(batch_size):
        scores = top_k_scores[i].tolist()
        assert scores == sorted(scores, reverse=True), f"分数未按降序排列: {scores}"

    print("✓ Top-K检索测试通过\n")


def test_top_k_with_invalid_filtering():
    """测试Top-K检索功能(带无效ID过滤)"""
    print("=" * 50)
    print("测试4: Top-K检索(带无效ID过滤)")
    print("=" * 50)

    num_items = 50
    embedding_dim = 32
    batch_size = 3
    k = 5

    # 创建embedding模块
    embedding = LocalEmbedding(num_items=num_items, embedding_dim=embedding_dim)

    # 创建查询向量
    query_embeddings = torch.randn(batch_size, embedding_dim)

    # 创建无效ID列表(用户历史)
    invalid_ids = torch.tensor([[1, 2, 3], [5, 10, 15], [20, 25, 30]])

    print(f"查询向量 shape: {query_embeddings.shape}")
    print(f"无效ID: {invalid_ids.tolist()}")
    print(f"检索Top-{k}(过滤后)")

    # 执行检索
    top_k_ids, top_k_scores = embedding.get_top_k_outputs(
        query_embeddings=query_embeddings, k=k, invalid_ids=invalid_ids
    )

    print(f"\n✓ Top-K IDs shape: {top_k_ids.shape}")
    print(f"✓ Top-K scores shape: {top_k_scores.shape}")

    # 验证没有无效ID出现在结果中
    for i in range(batch_size):
        result_ids = top_k_ids[i].tolist()
        invalid_list = invalid_ids[i].tolist()
        for item_id in result_ids:
            assert item_id not in invalid_list, f"无效ID {item_id} 出现在结果中!"
        print(f"✓ Batch {i}: 结果 {result_ids[:3]}... (无无效ID)")

    print("✓ 带过滤的Top-K检索测试通过\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "#" * 50)
    print("# 开始运行嵌入模块测试")
    print("#" * 50)

    try:
        test_local_embedding()
        test_filter_invalid_ids()
        test_top_k_retrieval()
        test_top_k_with_invalid_filtering()

        print("\n" + "#" * 50)
        print("# ✅ 所有测试通过!")
        print("#" * 50 + "\n")

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ 运行错误: {e}\n")
        raise


if __name__ == "__main__":
    run_all_tests()

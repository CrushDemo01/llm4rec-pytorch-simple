import os
from typing import List, Optional, Tuple

import hydra
import lightning as L
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from llm4rec_pytorch_simple.utils.pylogger import RankedLogger

logger = RankedLogger(__name__)  # __name__ 的值为模块的实际名称（包路径）


def eval_str2list(s: str, ignore_last_n: int = 0):
    y = [int(x) for x in s.split(",")]
    if ignore_last_n > 0:
        y = y[:-ignore_last_n]
    return y


def eval_int_list(
    x,
    target_len: int,
    ignore_last_n: int,
    shift_id_by: int,
    sampling_kept_mask: Optional[List[bool]] = None,
) -> Tuple[List[int], int]:
    """
    处理整数列表，包括忽略元素、ID偏移、采样和序列反转

    参数:
        x: 字符串形式的列表
        target_len: 目标长度
        ignore_last_n: 要忽略的元素数量
        shift_id_by: ID偏移量
        sampling_kept_mask: 采样掩码，指示哪些元素被保留

    返回:
        处理后的整数列表和列表长度
    """
    y = eval_str2list(x, ignore_last_n=ignore_last_n)
    if sampling_kept_mask is not None:
        y = [x for x, kept in zip(y, sampling_kept_mask) if kept]
    y_len = len(y)
    y.reverse()  # 反转序列，使最新交互在前. 这样后面取的时候直接 0 就是第一个了
    if shift_id_by > 0:
        y = [x + shift_id_by for x in y]
    return y, y_len


def truncate_or_pad_seq(y: List[int], target_len: int, chronological: bool) -> List[int]:
    """
    截断或填充序列到目标长度

    参数:
        y: 输入序列
        target_len: 目标长度
        chronological: 是否按时间顺序排列

    返回:
        处理后的序列
    """
    y_len = len(y)
    if y_len < target_len:
        # 序列太短，用0填充
        y = y + [0] * (target_len - y_len)
    else:
        # 序列太长，需要截断
        if not chronological:
            # 逆序：保留前面的元素(最新的交互)
            y = y[:target_len]
        else:
            # 时间顺序：保留后面的元素(最新的交互)
            y = y[-target_len:]
    assert len(y) == target_len
    return y


class RecDataset(Dataset):
    def __init__(
        self,
        ratings_file: str,
        max_sequence_length: int,
        ignore_last_n: int,
        shift_id_by: int = 0,
        chronological: bool = False,
        sample_ratio: float = 1.0,
    ):
        """
        参数:
            ratings_file: 评分文件路径或DataFrame，包含用户-物品交互数据
            padding_length(max_sequence_length): 序列填充的目标长度，所有序列将被填充或截断到此长度
            ignore_last_n: 忽略最后的交互数量，用于创建训练/验证/测试集
                          (例如，对于训练集，ignore_last_n=1表示忽略最后一个交互作为预测目标)
            shift_id_by: ID偏移量，用于调整物品或用户ID的起始值，默认为0
            chronological: 是否按时间顺序排列交互，默认为False(逆序，最新交互在前)
            sample_ratio: 采样比例，用于减少训练数据量，默认为1.0(使用全部数据)
        """
        super().__init__()
        logger.info(f"初始化 RecDataset，数据文件: {os.path.abspath(ratings_file)}")
        self.ratings_frame: pd.DataFrame = pd.read_csv(ratings_file, delimiter=",")
        self.padding_length = max_sequence_length
        self.ignore_last_n = ignore_last_n
        self.shift_id_by = shift_id_by
        self.chronological = chronological
        self.sample_ratio = sample_ratio
        self._cache = dict()  # 缓存已处理的样本，提高重复访问效率

    def __len__(self):
        return len(self.ratings_frame)

    def __getitem__(self, index: int):
        """
        获取指定索引的样本
        """
        if index in self._cache:
            return self._cache[index]

        # 使用 __load_data 方法获取完整的处理后数据
        data_dict = self.__load_data(index)
        self._cache[index] = data_dict
        return data_dict

    def __load_data(self, idx: int):
        data = self.ratings_frame.iloc[idx]

        # 采样
        sampling_kept_mask = None
        if self.sample_ratio < 1.0:
            sequence_item_ids_ = eval_str2list(data["sequence_item_ids"], self.ignore_last_n)
            raw_length = len(sequence_item_ids_)
            sampling_kept_mask = (torch.rand(raw_length) < self.sample_ratio).tolist()

        # 处理物品 id：包括转成 list、ID 偏移、采样、反转（新的在前面）
        movie_history, movie_history_len = eval_int_list(
            data["sequence_item_ids"],
            self.padding_length,
            self.ignore_last_n,
            self.shift_id_by,  # 只有id 可能偏移一下
            sampling_kept_mask,
        )
        # 处理评分序列
        movie_history_ratings, ratings_len = eval_int_list(
            data["sequence_ratings"],
            self.padding_length,
            self.ignore_last_n,
            0,
            sampling_kept_mask=sampling_kept_mask,
        )
        # 处理时间戳序列
        movie_timestamps, timestamps_len = eval_int_list(
            data["sequence_timestamps"],
            self.padding_length,
            self.ignore_last_n,
            0,
            sampling_kept_mask=sampling_kept_mask,
        )
        # 确保所有序列长度一致
        assert movie_history_len == timestamps_len, (
            f"history len {movie_history_len} differs from timestamp len {timestamps_len}."
        )
        assert movie_history_len == ratings_len, (
            f"history len {movie_history_len} differs from ratings len {ratings_len}."
        )

        # 分离历史序列和目标(预测目标)
        # 第一个元素(最新的交互)作为目标，其余作为历史
        historical_ids = movie_history[1:]
        historical_ratings = movie_history_ratings[1:]
        historical_timestamps = movie_timestamps[1:]
        target_ids = movie_history[0]
        target_ratings = movie_history_ratings[0]
        target_timestamps = movie_timestamps[0]

        # 如果需要按时间顺序排列，则反转历史序列
        if self.chronological:
            historical_ids.reverse()
            historical_ratings.reverse()
            historical_timestamps.reverse()

        # 处理序列长度，确保不超过最大长度
        max_seq_len = self.padding_length - 1  # 这个是历史序列长度
        history_length = min(len(historical_ids), max_seq_len)

        historical_ids = truncate_or_pad_seq(
            historical_ids,
            max_seq_len,
            self.chronological,
        )
        historical_ratings = truncate_or_pad_seq(
            historical_ratings,
            max_seq_len,
            self.chronological,
        )
        historical_timestamps = truncate_or_pad_seq(
            historical_timestamps,
            max_seq_len,
            self.chronological,
        )

        # 构建返回结果
        ret = {
            "user_id": data["user_id"],
            "historical_ids": torch.tensor(historical_ids, dtype=torch.int64),
            "historical_ratings": torch.tensor(historical_ratings, dtype=torch.int64),
            "historical_timestamps": torch.tensor(historical_timestamps, dtype=torch.int64),
            "history_lengths": history_length,
            "target_ids": target_ids,
            "target_ratings": target_ratings,
            "target_timestamps": target_timestamps,
        }
        return ret


class RecoDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset: DictConfig = None,
        val_dataset: DictConfig = None,
        test_dataset: DictConfig = None,
        max_hash_ranges: dict = {"genres": 63, "title": 16383, "year": 511},
        data_path: str = "ml-1m/data",
        max_jagged_dimension: int = 16,
        max_sequence_length: int = 200,
        chronological: bool = True,
        batch_size: int = 4,
        sample_ratio: float = 1.0,
        num_workers: int = os.cpu_count() // 4,
        prefetch_factor: int = 4,
    ):
        super().__init__()
        self.data_path = data_path
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.max_sequence_length = max_sequence_length
        self.chronological = chronological
        self.batch_size = batch_size
        self.sample_ratio = sample_ratio
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.max_jagged_dimension = max_jagged_dimension  # # 每个特征保留的最大数据
        self.max_hash_ranges = max_hash_ranges
        self.movie_feat = self.__init_item_ids()

    def __init_item_ids(self):
        """
        初始化物品ID和相关特征数据

        对于MovieLens数据集，该方法会处理物品特征(如电影类型、标题、年份等)，
        并将这些特征转换为可用于模型的数值表示。对于其他数据集，则简单地
        创建连续的物品ID列表。

        处理流程：
        1. 读取物品特征数据
        2. 对每个特征类型进行哈希处理
        3. 存储特征向量和长度信息
        4. 创建物品ID列表
        """
        logger.info(f"Initializing item IDs and features from {os.path.abspath(self.data_path)}")
        movies = pd.read_csv(
            f"{self.data_path}/movies.dat", sep="::", header=0, engine="python", names=["movie_id", "title", "genres"]
        )
        # 提取年份和清理标题
        movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")
        movies["clean_title"] = movies["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()

        # max_item_id = movies["movie_id"].max()
        # movie_id::title::genres
        movie_columns = ["genres", "title", "year"]

        logger.info(f"self.max_hash_ranges: {self.max_hash_ranges}")

        # 为movies DataFrame添加特征列
        for feature_name in movie_columns:
            movies[f"{feature_name}_feat_length"] = 0
            movies[f"{feature_name}_feat_value"] = ""  # 初始化为字符串类型以存储列表

        for idx, row in movies.iterrows():
            # movie_id = int(row["movie_id"])
            genres = row["genres"].split("|")  # 电影类型，可能有多个
            titles = row["clean_title"].split(" ")  # 清理后的标题词
            years = [row["year"]]  # 年份

            # 遍历特征做处理，主要是做hash
            for feature_name, feature in zip(movie_columns, [genres, titles, years]):
                feature_vector = [hash(val) % self.max_hash_ranges[feature_name] for val in feature]
                # 存储到DataFrame中
                movies.loc[idx, f"{feature_name}_feat_length"] = min(self.max_jagged_dimension, len(feature))
                # 存储特征向量，最多保留max_jagged_dimension个特征（转换为字符串）
                movies.loc[idx, f"{feature_name}_feat_value"] = str(feature_vector[: self.max_jagged_dimension])

        return movies

    def setup(self, stage: str = "test"):
        logger.info(f"Setting up dataset for stage: {stage}")
        """
        设置数据集，根据训练、验证或测试阶段加载数据
        """
        kwargs = {
            "max_sequence_length": self.max_sequence_length,
            "chronological": self.chronological,
            "sample_ratio": self.sample_ratio,
        }
        if stage == "fit":
            self.train_dataset = hydra.utils.instantiate(self.train_dataset, **kwargs)
            self.val_dataset = hydra.utils.instantiate(self.val_dataset, **kwargs)
        elif stage == "test":
            self.test_dataset = hydra.utils.instantiate(self.test_dataset, **kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def save_predictions(self, output_file: str, predictions: dict):
        """
        将预测结果保存到文件

        该方法将模型预测结果与测试数据集合并，并保存到指定文件。
        预测结果将被添加为新的列到原始测试数据中。

        参数:
            output_file: 输出文件路径，必须是CSV格式
            predictions: 预测结果字典，键为列名，值为预测值列表或数组
                       长度和顺序必须与测试数据集一致

        使用示例:
            >>> predictions = {"predicted_rating": [4, 3, 5, ...]}
            >>> datamodule.save_predictions("predictions.csv", predictions)
        """
        ratings_frame = self.test_dataset.ratings_frame
        # 将预测结果添加到测试数据框中
        for key, value in predictions.items():
            ratings_frame[key] = value
        # 保存合并后的数据到文件
        ratings_frame.to_csv(output_file, index=False)


# 你直接运行了 rec_dataset.py，它位于 data/ 目录下，而 Hydra 总是把 config_path 基于“执行文件所在目录”作为默认 provider root，这导致 config 搜索路径完全错误。
@hydra.main(version_base="1.1", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    cfg = cfg.data
    print(f"cfg: {cfg}")
    # 2. 创建RecoDataModule实例
    kwargs = {
        "train_dataset": cfg.train_dataset,
        "val_dataset": cfg.val_dataset,
        "test_dataset": cfg.test_dataset,
    }
    datamodule: RecoDataModule = hydra.utils.instantiate(cfg.data_module, **kwargs, _recursive_=False)

    print("✓ RecoDataModule 实例创建成功")

    # 3. 测试电影特征初始化
    print("\n=== 电影特征数据测试 ===")
    movie_feat = datamodule.movie_feat
    print(f"电影特征DataFrame形状: {movie_feat.shape}")
    print(f"电影特征列: {list(movie_feat.columns)}")

    # 显示前5部电影的特征信息
    print("\n前5部电影的特征信息:")
    for i in range(min(5, len(movie_feat))):
        row = movie_feat.iloc[i]
        print(f"电影 {i + 1}: ID={row['movie_id']}, 标题='{row['clean_title'][:30]}...'")
        print(f"  类型特征: 长度={row['genres_feat_length']}, 值={row['genres_feat_value']}")
        print(f"  标题特征: 长度={row['title_feat_length']}, 值={row['title_feat_value']}")
        print(f"  年份特征: 长度={row['year_feat_length']}, 值={row['year_feat_value']}")
        print()

    # 4. 测试数据加载器设置
    print("=== 数据加载器测试 ===")
    datamodule.setup(stage="test")
    print("✓ 测试数据集设置完成")

    # 5. 创建测试数据加载器
    test_loader = datamodule.test_dataloader()
    print(f"✓ 测试数据加载器创建成功，批量大小: {test_loader.batch_size}")
    print(f"✓ 数据加载器中的样本数量: {len(test_loader.dataset)}")

    # 6. 测试批次加载
    print("\n=== 批次数据加载测试 ===")
    for batch_idx, batch in enumerate(test_loader):
        print(f"批次 {batch_idx + 1}:")
        print(f"  用户ID: {batch['user_id']}")
        print(f"  历史物品ID形状: {batch['historical_ids'].shape}")
        print(f"  历史评分形状: {batch['historical_ratings'].shape}")
        print(f"  历史时间戳形状: {batch['historical_timestamps'].shape}")
        print(f"  历史长度: {batch['history_lengths']}")
        print(f"  目标物品ID: {batch['target_ids']}")
        print(f"  目标评分: {batch['target_ratings']}")
        print(f"  目标时间戳: {batch['target_timestamps']}")

        # 只测试前2个批次
        if batch_idx >= 1:
            break

    print("\n=== 测试预测结果保存功能 ===")
    # 创建模拟预测结果
    num_samples = len(test_loader.dataset)
    mock_predictions = {"predicted_rating": [3.5] * num_samples, "predicted_rank": list(range(1, num_samples + 1))}

    # 保存预测结果
    output_file = "test_predictions.csv"
    datamodule.save_predictions(output_file, mock_predictions)
    print(f"✓ 预测结果已保存到: {output_file}")

    print("\n=== RecoDataModule 测试完成 ===")


if __name__ == "__main__":
    main()

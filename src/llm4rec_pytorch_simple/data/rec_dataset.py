import os
from typing import Optional, List

import hydra
import lightning as L
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
import os
import ast
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from llm4rec_pytorch_simple.utils.pylogger import RankedLogger

logger = RankedLogger(__name__)  # __name__ 的值为模块的实际名称（包路径）

class RecoDataset(torch.utils.data.Dataset):
    """
    优化后的序列推荐数据集类。
    """

    def __init__(
        self,
        ratings_file: str,
        max_sequence_length: int,
        ignore_last_n: int,
        sample_ratio: float = 1.0,
        # 【优化 4】Python 陷阱：修复可变默认参数，使用 None 替代 []
        additional_columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        logger.info(f"初始化 RecDataset，数据文件: {os.path.abspath(ratings_file)}")
        self.ratings_frame: pd.DataFrame = pd.read_csv(ratings_file, delimiter=",")
        self.max_sequence_length = max_sequence_length
        self.ignore_last_n = ignore_last_n
        self.sample_ratio = sample_ratio
        
        # 初始化列表
        self.additional_columns = additional_columns if additional_columns is not None else []# 检查列是否存在
        self._check_columns()
        
        # 一次性eval进内容（小数据量）
        for col in ['sequence_item_ids', 'sequence_ratings', 'sequence_timestamps']:
            if isinstance(self.ratings_frame[col].iloc[0], str):
                 # 使用 ast.literal_eval 更安全
                self.ratings_frame[col] = self.ratings_frame[col].apply(ast.literal_eval)
    
    def _check_columns(self):
        """检查必要的列是否存在"""
        required_columns = ['user_id', 'sequence_item_ids', 'sequence_ratings', 'sequence_timestamps']
        for col in required_columns + self.additional_columns:
            if col not in self.ratings_frame.columns:
                raise ValueError(f"Column {col} does not exist in the ratings data.")
            
    def __len__(self) -> int:
        return len(self.ratings_frame)
    
    def _process_sequence(
        self, 
        seq_list: List,
        sampling_kept_mask: Optional[list[bool]] = None,
    ) -> Tuple[torch.Tensor, Any, int]:
        """
        处理单个序列：截断 -> 分割 History/Target -> Padding
        """
        # A. Ignore Last N (用于构建训练/验证集)
        if self.ignore_last_n > 0:
            seq_list = seq_list[:-self.ignore_last_n]
        
        # B. 应用 mask 采样
        if sampling_kept_mask is not None:
            seq_list = [item for item, keep in zip(seq_list, sampling_kept_mask) if keep]
        
        # C. 处理空序列
        if len(seq_list) == 0:
            return torch.zeros(self.max_sequence_length, dtype=torch.int64), 0, 0
        
        # D. 分割 Target 和 History， 历史序列截断到 max_sequence_length
        # 假设 seq_list 是按时间正序排列 [1, 2, 3, 4]
        target = seq_list[-1]      # 4
        history = seq_list[:-1][-self.max_sequence_length:]    # [1, 2, 3]
        history_length = len(history)
        
        # 对不足 max_sequence_length 的历史序列进行 Padding# 创建全 0 Tensor
        padding_length = self.max_sequence_length - history_length
        if padding_length > 0:
            padding_seq = torch.zeros(padding_length, dtype=torch.int64)
            history_tensor = torch.cat([torch.tensor(history, dtype=torch.int64), padding_seq]) if len(history) > 0 else padding_seq
        else:
            history_tensor = torch.tensor(history, dtype=torch.int64)
            
        return history_tensor, torch.tensor(target, dtype=torch.int64), history_length

    def __getitem__(self, idx) -> Dict[str, Any]:
        # 直接获取预处理好的 List 数据
        data = self.ratings_frame.iloc[idx]
        
        # 采样
        sampling_kept_mask = None
        if self.sample_ratio < 1.0:
            raw_length = len(data["sequence_item_ids"]) - self.ignore_last_n
            sampling_kept_mask = (torch.rand(raw_length) < self.sample_ratio).tolist()
        
        historical_ids, target_ids, history_ids_length = self._process_sequence(data["sequence_item_ids"], sampling_kept_mask)
        historical_ratings, target_ratings, history_ratings_length = self._process_sequence(data["sequence_ratings"], sampling_kept_mask)
        historical_timestamps, target_timestamps, history_timestamps_length = self._process_sequence(data["sequence_timestamps"], sampling_kept_mask)
        # 确保所有序列长度一致
        assert history_ids_length == history_timestamps_length, (
            f"history len {history_ids_length} differs from timestamp len {history_timestamps_length}."
        )
        assert history_ids_length == history_ratings_length, (
            f"history len {history_ids_length} differs from ratings len {history_ratings_length}."
        )
        
        # 构建返回结果
        ret = {
            "user_id": data["user_id"],
            "historical_ids": historical_ids,
            "historical_ratings": historical_ratings,
            "historical_timestamps": historical_timestamps,
            "history_lengths": history_ids_length,
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
        max_hash_ranges: Optional[dict] = None,
        data_path: str = "ml-1m/data",
        max_jagged_dimension: int = 16,
        max_sequence_length: int = 200,
        batch_size: int = 4,
        sample_ratio: float = 1.0,
        num_workers: int = os.cpu_count() // 4,
        prefetch_factor: int = 4,
        persistent_workers: bool = False,
        **kwargs,
    ):
        if max_hash_ranges is None:
            max_hash_ranges = {"genres": 63, "title": 16383, "year": 511}
        super().__init__()
        self.data_path = data_path
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.sample_ratio = sample_ratio
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
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
        设置数据集，根据训练、验证或测试阶段加载数据，在fit, test, predict 等都会自动调用
        """
        kwargs = {
            "max_sequence_length": self.max_sequence_length,
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
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
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
@hydra.main(version_base="1.2", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    cfg = cfg.data
    print("=== RecoDataModule 测试 ===")
    print(f"cfg: {cfg}")
    # 2. 创建RecoDataModule实例
    kwargs = {
        "train_dataset": cfg.train_dataset,
        "val_dataset": cfg.val_dataset,
        "test_dataset": cfg.test_dataset,
    }
    datamodule: RecoDataModule = hydra.utils.instantiate(cfg, **kwargs, _recursive_=False)

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

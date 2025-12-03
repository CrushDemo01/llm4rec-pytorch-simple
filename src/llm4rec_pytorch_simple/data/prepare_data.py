"""
处理数据，将原始数据处理为模型输入的格式
参考https://raw.githubusercontent.com/kang205/SASRec的格式
例如：
index,user_id,sequence_item_ids,sequence_ratings,sequence_timestamps,sex,age_group,occupation,zip_code
0,1,1193:661:914:3408:2355:1197:1287:2804:594:919,5:3:3:4:5:3:5:5:4:4,978300760:978302109:978301968:978300275:978824291:978302268:978302039:978300719:978302268:978301368,F,1,10,48067
"""

import os

os.sys.path.append("./src/llm4rec_pytorch_simple")
# 从 hydra 配置中获取各种参数

import json

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from llm4rec_pytorch_simple.utils.pylogger import RankedLogger

logger = RankedLogger(__name__)


class PrepareDataMovieLens:
    def __init__(self, data_dir: str, max_hash_ranges: dict, **kwargs):
        assert "ml-1m" in data_dir, "暂时只支持处理MovieLens 1M数据集"
        self.data_dir = os.path.abspath(data_dir)
        self.processed_dir = os.path.abspath(os.path.join(self.data_dir, "../", "processed"))
        os.makedirs(self.processed_dir, exist_ok=True)
        self.max_hash_ranges = max_hash_ranges
        self.users_info = None

        # 创建 movie_id 到连续索引的映射
        self.movie_id_mapping, self.idx_to_movie_id = self._create_movie_id_mapping()
        logger.info(f"原始目录：{self.data_dir}，输出目录：{self.processed_dir}")
        logger.info(f"Movie ID 映射创建完成: {len(self.movie_id_mapping)} 部电影")

    def _create_movie_id_mapping(self):
        """
        创建 movie_id 到连续索引的映射
        返回: (movie_id -> index, index -> movie_id)
        """
        movies_file = os.path.join(self.data_dir, "movies.dat")
        movies = pd.read_csv(movies_file, sep="::", header=0, engine="python", names=["movie_id", "title", "genres"])

        # 获取所有唯一的 movie_id 并排序
        unique_movie_ids = sorted(movies["movie_id"].unique())

        # 创建映射: movie_id -> sequential_index (从 1 开始,0 保留给 padding)
        movie_id_to_idx = {int(movie_id): idx + 1 for idx, movie_id in enumerate(unique_movie_ids)}
        idx_to_movie_id = {idx + 1: int(movie_id) for idx, movie_id in enumerate(unique_movie_ids)}

        # 保存映射文件
        mapping_file = os.path.join(self.processed_dir, "movie_id_mapping.json")
        with open(mapping_file, "w") as f:
            json.dump(
                {
                    "movie_id_to_idx": {str(k): v for k, v in movie_id_to_idx.items()},  # 键转为字符串
                    "idx_to_movie_id": {str(k): v for k, v in idx_to_movie_id.items()},  # 键转为字符串
                    "num_movies": len(unique_movie_ids),
                    "max_original_id": int(max(unique_movie_ids)),
                    "max_mapped_idx": len(unique_movie_ids),
                },
                f,
                indent=2,
            )

        logger.info(f"Movie ID 映射已保存到: {mapping_file}")
        logger.info(f"原始 movie_id 范围: 1-{max(unique_movie_ids)}, 映射后索引范围: 1-{len(unique_movie_ids)}")

        return movie_id_to_idx, idx_to_movie_id

    def _split_by_user_temporal(self, ratings_df):
        """
        方式一：用户级时间划分
        对每个用户，按照时间排序，前80%交互作为训练集，后20%作为测试集。
        优点：保证每个用户都在测试集中出现，适合个性化推荐评估。
        """
        logger.info("执行用户级时间划分（每个用户 前80% Train / 后20% Test）...")

        train_data_list = []
        test_data_list = []

        # 按用户分组处理
        # sort_values 在外层做一次比在循环里做效率略高
        sorted_ratings = ratings_df.sort_values(by=["user_id", "timestamp"])

        for _user_id, user_ratings in sorted_ratings.groupby("user_id"):
            user_ratings_list = user_ratings.values.tolist()
            total_interactions = len(user_ratings_list)

            split_point = int(total_interactions * 0.8)

            if split_point > 0:
                train_data_list.extend(user_ratings_list[:split_point])

            if split_point < total_interactions:
                test_data_list.extend(user_ratings_list[split_point:])

        columns = ["user_id", "movie_id", "rating", "timestamp"]
        return pd.DataFrame(train_data_list, columns=columns), pd.DataFrame(test_data_list, columns=columns)

    def _split_by_global_temporal(self, ratings_df):
        """
        方式二：全局时间划分
        不区分用户，将所有交互记录按时间排序，前80%的时间段数据作为训练集，后20%作为测试集。
        优点：模拟真实的未来预测场景，严防数据穿越。
        缺点：可能会导致测试集中出现训练集中从未见过的冷启动用户。
        """
        logger.info("执行全局时间划分（整体时间线 前80% Train / 后20% Test）...")

        # 1. 全局按时间戳排序
        sorted_ratings = ratings_df.sort_values(by="timestamp")

        # 2. 计算切分索引
        total_len = len(sorted_ratings)
        split_index = int(total_len * 0.8)

        # 3. 切分 DataFrame
        train_df = sorted_ratings.iloc[:split_index].copy()
        test_df = sorted_ratings.iloc[split_index:].copy()

        # 打印一下切分的时间点供参考
        split_time = pd.to_datetime(train_df.iloc[-1]["timestamp"], unit="s")
        logger.info(f"全局切分时间点约为：{split_time} (Index: {split_index})")

        return train_df, test_df

    def _split_by_user_id(self, ratings_df):
        """
        方式三：按用户ID划分
        将用户按ID排序，前80%的用户的所有交互作为训练集，后20%的用户的所有交互作为测试集。
        优点：评估模型对新用户（冷启动）的泛化能力。
        """
        logger.info("执行按用户ID划分（User ID 前80% Train / 后20% Test）...")

        user_ids = sorted(ratings_df["user_id"].unique())
        total_users = len(user_ids)
        split_idx = int(total_users * 0.8)

        train_user_ids = set(user_ids[:split_idx])
        test_user_ids = set(user_ids[split_idx:])

        train_df = ratings_df[ratings_df["user_id"].isin(train_user_ids)]
        test_df = ratings_df[ratings_df["user_id"].isin(test_user_ids)]

        return train_df, test_df

    def _create_sequential_data(self, df, data_type):
        """
        辅助函数：将 Rating DataFrame 转换为 SASRec 需要的序列格式并保存
        应用 movie_id 映射,将原始 movie_id 转换为连续索引
        """
        if df.empty:
            logger.warning(f"{data_type} 数据集为空，跳过生成序列文件。")
            return

        # 应用 movie_id 映射
        df["movie_id"] = df["movie_id"].map(self.movie_id_mapping)

        # 检查是否有未映射的 movie_id
        if df["movie_id"].isna().any():
            unmapped_count = df["movie_id"].isna().sum()
            logger.warning(f"发现 {unmapped_count} 个未映射的 movie_id,将被过滤掉")
            df = df.dropna(subset=["movie_id"])

        # 转换为整数
        df["movie_id"] = df["movie_id"].astype(int)

        # 按时间戳排序并按用户分组
        df_grouped = df.sort_values(by=["timestamp"]).groupby("user_id")

        seq_data = pd.DataFrame(
            data={
                "user_id": list(df_grouped.groups.keys()),
                "sequence_item_ids": list(df_grouped["movie_id"].apply(lambda x: ",".join(map(str, x)))),
                "sequence_ratings": list(df_grouped["rating"].apply(lambda x: ",".join(map(str, x)))),
                "sequence_timestamps": list(df_grouped["timestamp"].apply(lambda x: ",".join(map(str, x)))),
            }
        )

        # merge 用户特征数据
        if self.users_info is not None:
            seq_data = pd.merge(seq_data, self.users_info, on="user_id", how="left")

        logger.info(f"{data_type}集序列构建完成，用户数：{len(seq_data)}")

        # 保存
        output_path = os.path.join(self.processed_dir, f"sasrec_format_{data_type.lower()}_data.csv")
        seq_data.reset_index(drop=True).to_csv(output_path, index=False)
        return seq_data

    def prepare(self, split_mode: str = "user"):
        """
        数据准备主入口
        :param split_mode: 划分模式, "user" (用户级时间划分) 或 "global" (全局时间划分) 或 "user_id" (按用户ID划分)
        """
        assert "ml-1m" in self.data_dir, "只支持处理MovieLens 1M数据集"
        logger.info(
            f"当前划分模式：{split_mode}; 原始目录：{os.path.abspath(self.data_dir)}，输出目录：{os.path.abspath(self.processed_dir)}"
        )
        # --- 1. 加载数据 ---
        users = pd.read_csv(
            os.path.join(self.data_dir, "users.dat"),
            sep=r"::",
            header=0,
            engine="python",
            names=["user_id", "sex", "age_group", "occupation", "zip_code"],
        )

        movies = pd.read_csv(
            os.path.join(self.data_dir, "movies.dat"),
            sep=r"::",
            header=0,
            engine="python",
            names=["movie_id", "title", "genres"],
        )

        ratings = pd.read_csv(
            os.path.join(self.data_dir, "ratings.dat"),
            sep=r"::",
            header=0,
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
        )

        # --- 2. 数据预处理 ---
        # 这里不处理movie的数据，而是在dataset 初始化的啥时候在处理movie的特征

        # 特征编码
        users["sex"] = pd.Categorical(users["sex"]).codes
        users["age_group"] = pd.Categorical(users["age_group"]).codes
        users["occupation"] = pd.Categorical(users["occupation"]).codes
        users["zip_code"] = pd.Categorical(users["zip_code"]).codes

        # 保存 users 信息供后续 merge 使用
        self.users_info = users

        logger.info(
            f"用户数：{users['user_id'].nunique()}，物品数：{movies['movie_id'].nunique()}，评分数：{len(ratings)}"
        )

        # --- 3. 划分训练集和测试集 ---
        if split_mode == "user":
            train_df, test_df = self._split_by_user_temporal(ratings)
        elif split_mode == "global":
            train_df, test_df = self._split_by_global_temporal(ratings)
        elif split_mode == "user_id":
            train_df, test_df = self._split_by_user_id(ratings)
        else:
            raise ValueError(f"不支持的划分模式: {split_mode}, 请使用 'user', 'global' 或 'user_id'")

        logger.info(f"训练集大小：{len(train_df)}，测试集大小：{len(test_df)}")

        # --- 4. 创建序列格式并保存 ---
        self._create_sequential_data(train_df, f"{split_mode}_train")
        self._create_sequential_data(test_df, f"{split_mode}_test")

        logger.info(f"处理完成，文件保存在：{self.processed_dir}")


@hydra.main(version_base="1.1", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    logger.info(f"{OmegaConf.to_yaml(cfg.data.data_prepare)}")
    # logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    preprocessor: PrepareDataMovieLens = hydra.utils.instantiate(cfg.data.data_prepare)

    # 方式一：原有逻辑（用户内划分）
    preprocessor.prepare(split_mode="user")
    # 方式二：新逻辑（全局时间划分）
    preprocessor.prepare(split_mode="global")
    # 方式三：按用户ID划分
    preprocessor.prepare(split_mode="user_id")


if __name__ == "__main__":
    main()

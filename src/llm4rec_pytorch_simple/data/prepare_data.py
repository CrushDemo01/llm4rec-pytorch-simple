"""
处理数据，将原始数据处理为模型输入的格式
参考https://raw.githubusercontent.com/kang205/SASRec的格式
例如：
index,user_id,sequence_item_ids,sequence_ratings,sequence_timestamps,sex,age_group,occupation,zip_code
0,1,"3186,1721,1270,1022,2340,1836,3408,1207,2804,260,720,1193,919,608,2692,1961,2028,3105,938,1035,1962,1028,2018,150,1097,914,1287,2797,1246,2762,661,2918,531,3114,2791,1029,2321,1197,594,2398,1545,527,745,595,588,1,2687,783,2294,2355,1907,1566,48","4,4,5,5,3,5,4,4,5,4,3,5,4,4,4,5,5,5,4,5,4,5,4,5,4,3,5,4,4,4,3,4,4,4,4,5,3,3,4,4,4,5,3,5,4,5,3,4,4,5,4,4,5","978300019,978300055,978300055,978300055,978300103,978300172,978300275,978300719,978300719,978300760,978300760,978300760,978301368,978301398,978301570,978301590,978301619,978301713,978301752,978301753,978301753,978301777,978301777,978301777,978301953,978301968,978302039,978302039,978302091,978302091,978302109,978302124,978302149,978302174,978302188,978302205,978302205,978302268,978302268,978302281,978824139,978824195,978824268,978824268,978824268,978824268,978824268,978824291,978824291,978824291,978824330,978824330,978824351",0,0,10,1588
"""

import os

os.sys.path.append("./src/llm4rec_pytorch_simple")
# 从 hydra 配置中获取各种参数
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from llm4rec_pytorch_simple.utils.pylogger import RankedLogger

logger = RankedLogger(__name__)


class PrepareDataMovieLens:
    def __init__(self, data_dir: str):
        assert "ml-1m" in data_dir, "暂时只支持处理MovieLens 1M数据集"
        self.data_dir = os.path.abspath(data_dir)
        self.processed_dir = os.path.abspath(os.path.join(self.data_dir, "../", "processed"))
        os.makedirs(self.processed_dir, exist_ok=True)  # 加载用户数据的辅助变量，供 create_sequential_data 使用
        self.users_info = None
        logger.info(f"原始目录：{self.data_dir}，输出目录：{self.processed_dir}")

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

        for user_id, user_ratings in sorted_ratings.groupby("user_id"):
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

    def _create_sequential_data(self, df, data_type):
        """
        辅助函数：将 Rating DataFrame 转换为 SASRec 需要的序列格式并保存
        """
        if df.empty:
            logger.warning(f"{data_type} 数据集为空，跳过生成序列文件。")
            return

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
        :param split_mode: 划分模式, "user" (用户级时间划分) 或 "global" (全局时间划分)
        """
        assert "ml-1m" in self.data_dir, "只支持处理MovieLens 1M数据集"
        logger.info(
            f"当前划分模式：{split_mode}; 原始目录：{os.path.abspath(self.data_dir)}，输出目录：{os.path.abspath(self.processed_dir)}"
        )
        # --- 1. 加载数据 ---
        users = pd.read_csv(
            os.path.join(self.data_dir, "users.dat"),
            sep=r"::",
            header=None,
            engine="python",
            names=["user_id", "sex", "age_group", "occupation", "zip_code"],
        )

        movies = pd.read_csv(
            os.path.join(self.data_dir, "movies.dat"),
            sep=r"::",
            header=None,
            engine="python",
            names=["movie_id", "title", "genres"],
        )

        ratings = pd.read_csv(
            os.path.join(self.data_dir, "ratings.dat"),
            sep=r"::",
            header=None,
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
        )

        # --- 2. 数据预处理 ---
        # 提取年份和清理标题
        movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")
        movies["clean_title"] = movies["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()

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
        else:
            raise ValueError(f"不支持的划分模式: {split_mode}, 请使用 'user' 或 'global'")

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

    logger.info(f"{OmegaConf.to_yaml(cfg.data.data_preprocessor)}")
    # logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    preprocessor: PrepareDataMovieLens = hydra.utils.instantiate(cfg.data.data_preprocessor)

    # 方式一：原有逻辑（用户内划分）
    preprocessor.prepare(split_mode="user")
    # 方式二：新逻辑（全局时间划分）
    preprocessor.prepare(split_mode="global")


if __name__ == "__main__":
    main()

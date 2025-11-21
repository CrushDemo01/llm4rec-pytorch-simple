# 数据处理
推荐系统中的数据一般可以简单的分为两种类型：
- spares 特征：一般是用户或者物品的一些属性，比如用户的性别，物品的类别等。
- dense 特征：一般是用户或者物品的一些数值特征，比如用户的年龄，物品的价格等。

## sparse 和 dense 特征
### sparse 特征
sparse 特征 一般都会用 embedding 的形式，但也存在部分特征类别特征是直接用 one-hot 编码的情况。
> 但 one-hot 编码会导致维度灾难，特征向量就会非常大，且非常稀疏。即每个词都是一堆一堆一堆0中唯一的那个1。
我的体会是有全部使用 embedding 的倾向。

此外，对于一些 dense 特征(年龄，数量、点击数值和分位点等)，也有使用 embedding 的情况（分桶+分位点等方式）。这有利于表征更丰富的信息。

### dense 特征
dense 特征，大部分就是一些数值特征，比如用户的年龄，物品的价格， click 等。

### PS
在真的业务中，还会挖掘和人为划分各种特征。比如用户长短期兴趣，长短期行为等。

## MovieLens 数据集中的特征

MovieLens 数据集（以最经典的 **MovieLens-1M** 为例）虽然字段看似简单，但在构建精排模型（如 DeepFM, DIN, DCN）时，对特征的处理方式决定了模型的上限。

针对精排模型，我们通常不再直接使用原始数值，而是倾向于将绝大多数特征转化为 **Embedding 向量**。

以下是详细的特征处理方案对照表及深度解析：

### 1\. 特征处理方案总览表

| 特征域 | 原始字段名 | 原始类型 | 推荐处理方式 | 最终特征形式 (Input to Model) |
| :--- | :--- | :--- | :--- | :--- |
| **User** | `UserID` | ID / String | Label Encoding -\> Embedding | **Tensor (Batch, Embedding\_Dim)** |
| **User** | `Gender` | "F", "M" | Label Encoding -\> Embedding | **Tensor (Batch, Embedding\_Dim)** |
| **User** | `Age` | 数值/枚举 (1, 18, 25...) | **分桶 (Bucketing) / 离散化** -\> Embedding | **Tensor (Batch, Embedding\_Dim)** |
| **User** | `Occupation`| ID (0-20) | Label Encoding -\> Embedding | **Tensor (Batch, Embedding\_Dim)** |
| **User** | `Zip-code` | String | **Hash Encoding** (或丢弃) -\> Embedding | **Tensor (Batch, Embedding\_Dim)** |
| **Item** | `MovieID` | ID | Label Encoding -\> Embedding | **Tensor (Batch, Embedding\_Dim)** |
| **Item** | `Genres` | String ("Action|Sci-Fi") | **Multi-Hot Encoding** -\> Embedding Pooling | **Tensor (Batch, Embedding\_Dim)** |
| **Item** | `Title` | String | (通常忽略或用NLP提取) -\> Embedding | (基础模型中通常忽略) |
| **Context**| `Timestamp`| Long (1970s...) | **特征提取** (Hour, Weekday) -\> Embedding | **Tensor (Batch, Embedding\_Dim)** |

-----

### 2\. 详细特征工程解析

#### A. 简单稀疏特征 (Sparse Features)

**特征：** `UserID`, `MovieID`, `Gender`, `Occupation`
**特点：** 类别型数据，离散且无序。
**处理代码逻辑：**
这些是标准的 Embedding 处理流程。

1.  **构建字典 (Vocabulary):** 将所有出现的 ID 映射为 $0 \sim N-1$ 的整数。
2.  **PyTorch 实现:** 输入 LongTensor 索引，通过 `nn.Embedding`。

> **注意：** `Zip-code` 在 MovieLens 中非常稀疏且格式混乱，直接 Label Encoding 会导致字典过大且由大量低频词。**建议做法：** 使用 `Hash Encoding` (例如 hash 到 2000 个桶内) 或直接在前几位截断（保留区域信息）。

#### B. 数值特征离散化 (Numerical to Categorical)

**特征：** `Age`
**原始数据：** 在 ML-1M 中，Age 已经是离散的（1: "Under 18", 18: "18-24", 25: "25-34"...）。
**精排模型中的处理：**
虽然 Age 是数字，但在推荐系统中，**不要把它当作连续值归一化**。

  * **原因：** 年龄与看电影的喜好是非线性的。20岁和30岁的差异，并不等同于30岁和40岁的差异。
  * **最佳实践：** 把 `Age` 当作 **类别特征** 处理。
  * **操作：** 建立映射表 `{1: 0, 18: 1, 25: 2 ...}`，然后通过 `nn.Embedding` 学习每个年龄段的隐向量。

#### C. 多值离散特征 (Multi-Hot / Sequence Features)

**特征：** `Genres` (电影类别)
**原始数据：** "Animation|Children's|Comedy"
**特点：** 一部电影属于多个类别，长度不固定。
**最佳实践：** **Embedding Pooling (Sum/Mean)**

#### D. 时间特征 (Context Features)

**特征：** `Timestamp`
**原始数据：** 978300760 (Unix 时间戳)
**处理方式：** 原始数值无意义，需要挖掘。

1.  **绝对时间切片：** 转换为 `Hour_of_Day` (0-23), `Day_of_Week` (0-6)。转化为类别特征 -\> Embedding。
2.  **相对时间 (很重要)：** 计算 `Timestamp - User_First_Interaction_Time`。这代表了用户在这个平台活跃了多久，这类特征通常做归一化 (MinMax) 作为 Dense 特征，或者分桶后做 Embedding。

-----

### 3\. 进阶：用户行为序列 (User Behavior Sequence)

如果你要做 **DIN (Deep Interest Network)** 或 **SIM** 等精排模型，MovieLens 中最重要的“隐形特征”是**用户历史行为序列**。

**如何构造：**
利用 `Timestamp` 对用户的点击记录进行排序。

  * **Feature:** `User_History_Movie_IDs` (用户过去看过最近的 N 部电影 ID 列表)。
  * **Target Item:** `Target_Movie_ID` (当前要预测的电影)。
  * **处理方式：** Target Attention。计算 `Target_Movie_ID` 的 Embedding 与 `User_History` 中每个 Movie Embedding 的相似度，进行加权求和。

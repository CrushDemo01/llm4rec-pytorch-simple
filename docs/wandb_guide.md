# WandB 实验追踪使用指南

本项目已集成 Weights & Biases (WandB) 实验追踪平台，用于管理消融实验和对比不同配置。

## 快速开始

### 1. 首次使用：登录 WandB

```bash
wandb login
```

输入你的 API key（从 <https://wandb.ai/authorize> 获取）

### 2. 运行实验

**默认运行**（使用 WandB）：

```bash
uv run python src/llm4rec_pytorch_simple/scripts/train.py
```

**指定实验名称和标签**：

```bash
uv run python src/llm4rec_pytorch_simple/scripts/train.py \
  logger.wandb.name="SASRec-batch64" \
  logger.wandb.tags="[sasrec,batch_size_ablation]"
```

**离线模式**（稍后同步）：

```bash
uv run python src/llm4rec_pytorch_simple/scripts/train.py \
  logger.wandb.offline=True
```

## 消融实验示例

### 对比不同模型

```bash
# 实验 1: SASRec
uv run python src/llm4rec_pytorch_simple/scripts/train.py \
  model=SASRec \
  logger.wandb.tags="[ablation,model_comparison]" \
  logger.wandb.name="SASRec-baseline"

# 实验 2: HSTU
uv run python src/llm4rec_pytorch_simple/scripts/train.py \
  model=HSTU \
  logger.wandb.tags="[ablation,model_comparison]" \
  logger.wandb.name="HSTU-baseline"
```

### 对比不同超参数

```bash
# Batch Size 消融
uv run python src/llm4rec_pytorch_simple/scripts/train.py \
  data.batch_size=64 \
  logger.wandb.tags="[ablation,batch_size]" \
  logger.wandb.name="batch64"

uv run python src/llm4rec_pytorch_simple/scripts/train.py \
  data.batch_size=128 \
  logger.wandb.tags="[ablation,batch_size]" \
  logger.wandb.name="batch128"

# Learning Rate 消融
uv run python src/llm4rec_pytorch_simple/scripts/train.py \
  model.optimizer.lr=0.001 \
  logger.wandb.tags="[ablation,learning_rate]" \
  logger.wandb.name="lr_0.001"

uv run python src/llm4rec_pytorch_simple/scripts/train.py \
  model.optimizer.lr=0.0001 \
  logger.wandb.tags="[ablation,learning_rate]" \
  logger.wandb.name="lr_0.0001"
```

## 使用对比表格功能

这是 WandB 最强大的功能之一，可以一眼看出不同实验的参数差异。

### 步骤

1. **访问项目页面**：

   ```
   https://wandb.ai/your-username/llm4rec-pytorch-simple
   ```

2. **选择要对比的实验**：
   - 在 Runs 列表中勾选多个实验（按住 Shift 可批量选择）

3. **查看对比**：
   - 点击顶部的 **"Compare"** 按钮
   - 切换到 **"Config"** 标签页
   - WandB 会自动高亮显示不同的参数

4. **对比指标**：
   - 切换到 **"Charts"** 标签页
   - 查看训练曲线对比（Loss、HR@10、NDCG@10 等）

5. **使用配置表格**：
   - 在 **"Table"** 标签页查看所有实验的配置和指标
   - 可以按列排序、筛选

## 高级功能

### 平行坐标图

查看多个超参数与指标的关系：

1. 在项目页面点击 **"Parallel Coordinates"**
2. 选择要显示的参数（如 `batch_size`, `learning_rate`）
3. 选择要优化的指标（如 `val/hr@10`）
4. 一眼看出最佳参数组合

### 分组管理

实验会自动按模型类型分组（配置在 `wandb.yaml` 的 `group` 字段）：

- `llm4rec_pytorch_simple.models.archs.sasrec.SASRec`
- `llm4rec_pytorch_simple.models.archs.hstu.HSTU`

### 标签过滤

使用标签快速筛选实验：

```bash
# 添加多个标签
logger.wandb.tags="[ablation,batch_size,final_results]"
```

在 WandB 网页端点击标签即可过滤。

## 记录的内容

WandB 会自动记录：

### 超参数

- 模型配置（层数、隐藏维度、dropout 等）
- 数据配置（batch_size、数据集路径等）
- 训练配置（epochs、optimizer、scheduler 等）
- 模型参数量（总参数、可训练参数、不可训练参数）

### 指标

- 训练指标：`train/loss`, `train/hr@10` 等
- 验证指标：`val/loss`, `val/hr@10`, `val/ndcg@10` 等
- 测试指标：`test/hr@10`, `test/ndcg@10` 等

### 系统信息

- GPU 型号和数量
- Python 版本
- PyTorch 版本
- 主机名

### 模型文件

- 最佳模型检查点（如果启用 `log_model: True`）

## 切换回其他 Logger

如果需要临时使用其他 logger：

```bash
# 使用 CSV logger
uv run python src/llm4rec_pytorch_simple/scripts/train.py logger=csv

# 使用 TensorBoard
uv run python src/llm4rec_pytorch_simple/scripts/train.py logger=tensorboard

# 同时使用多个 logger
uv run python src/llm4rec_pytorch_simple/scripts/train.py logger=many_loggers
```

## 常见问题

### Q: 如何在离线环境使用？

A: 设置 `logger.wandb.offline=True`，稍后使用 `wandb sync` 同步。

### Q: 如何删除实验？

A: 在 WandB 网页端选中实验，点击 "Delete"。

### Q: 如何分享实验结果？

A: 在 WandB 网页端点击 "Share"，生成公开链接。

### Q: 实验记录保存多久？

A: 免费账户永久保存，无限制。

## 相关链接

- WandB 官方文档: <https://docs.wandb.ai/>
- WandB 注册: <https://wandb.ai/signup>
- API Key: <https://wandb.ai/authorize>

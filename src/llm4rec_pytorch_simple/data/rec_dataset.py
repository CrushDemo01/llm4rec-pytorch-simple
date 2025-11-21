from llm4rec_pytorch_simple.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)  # __name__ 的值为模块的实际名称（包路径）

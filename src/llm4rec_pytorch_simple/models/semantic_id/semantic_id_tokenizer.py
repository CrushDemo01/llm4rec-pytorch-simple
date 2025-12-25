"""
简化版语义ID Tokenizer

将物品 ID 转换为语义 ID 序列，支持双向映射
移除了 Trie 树和规则校验，只保留核心功能
"""
import torch
import pickle
from typing import Dict, List, Tuple, Optional, Union


class SemanticIdTokenizer:
    """
    简化版语义 ID Tokenizer

    核心功能：
    - item_id (int) ←→ semantic_ids (List[int])

    使用方式:
        # 从预构建的映射加载
        tokenizer = SemanticIdTokenizer.from_pretrained('semantic_id_mapping.pkl')

        # 使用
        semantic_ids = tokenizer.tokenize(item_ids)  # item → sem_id
        item_ids = tokenizer.decode(semantic_ids)     # sem_id → item
    """
    def __init__(self):
        self.item_id_to_semantic_ids: Dict[int, Tuple[int, ...]] = {}
        self.semantic_ids_to_item_id: Dict[Tuple[int, ...], int] = {}
        self.num_codebooks: Optional[int] = None
        self.vocab_size: Optional[int] = None
    
    @classmethod
    def from_pretrained(cls, mapping_path: str) -> 'SemanticIdTokenizer':
        """从预构建的映射加载"""
        tokenizer = cls()
        tokenizer.load_mapping(mapping_path)
        return tokenizer

    def load_mapping(self, path: str):
        """从映射文件加载"""
        print(f"从 {path} 加载映射...")
        with open(path, 'rb') as f:
            data = pickle.load(f)

        # 只加载必需的映射
        self.item_id_to_semantic_ids = data.get('item_id_to_semantic_ids', {})
        self.semantic_ids_to_item_id = data.get('semantic_ids_to_item_id', {})
        self.num_codebooks = data.get('num_codebooks')
        self.vocab_size = data.get('vocab_size')

        # 向后兼容：如果只有 Trie 树，重建映射（可选，根据需求可移除）
        if not self.semantic_ids_to_item_id and self.item_id_to_semantic_ids:
            self.semantic_ids_to_item_id = {
                tuple(v): k for k, v in self.item_id_to_semantic_ids.items()
            }

        print(f"映射已加载:")
        print(f"  - 物品总数: {len(self.item_id_to_semantic_ids)}")
        print(f"  - Codebook 数量: {self.num_codebooks}")
        print(f"  - 语义ID长度: {self.num_codebooks if self.num_codebooks else 'Unknown'}")

    def save_mapping(self, path: str):
        """保存映射到文件"""
        print(f"保存映射到 {path}...")
        data = {
            'item_id_to_semantic_ids': self.item_id_to_semantic_ids,
            'semantic_ids_to_item_id': self.semantic_ids_to_item_id,
            'num_codebooks': self.num_codebooks,
            'vocab_size': self.vocab_size,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print("保存完成")
    
    def tokenize(self, item_ids: Union[int, List[int]]) -> torch.Tensor:
        """
        将物品 ID 转换为语义 ID 序列
        
        参数:
            item_ids: 物品 ID (int 或 int 列表)
        返回:
            semantic_ids: Tensor，形状 (batch_size, num_codebooks+1)
            不存在的 ID 会被填充为 -1
        """
        # 处理不同输入格式
        if isinstance(item_ids, int):
            item_ids = [item_ids]
        
        # 批量转换
        semantic_ids_list = []
        for item_id in item_ids:
            if item_id in self.item_id_to_semantic_ids:
                semantic_ids_list.append(list(self.item_id_to_semantic_ids[item_id]))
            else:
                # 使用 -1 填充（不会与真实 semantic_ids 冲突）
                semantic_ids_list.append([-1] * self.num_codebooks)

        return torch.tensor(semantic_ids_list, dtype=torch.long)
    
    def decode(self, semantic_ids: torch.Tensor) -> List[Union[int, None]]:
        """
        将语义 ID 序列转换回物品 ID
        参数:
            semantic_ids: Tensor，形状 (batch_size, num_codebooks+1)
        返回:
            item_ids: 物品 ID 列表 (int)
        """
        result = []
        for sid_seq in semantic_ids:
            sid_tuple = tuple(sid_seq.tolist())
            if sid_tuple in self.semantic_ids_to_item_id:
                result.append(self.semantic_ids_to_item_id[sid_tuple])
            else:
                result.append(None)
        return result

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            'total_items': len(self.item_id_to_semantic_ids),
            'num_codebooks': self.num_codebooks,
            'semantic_id_length': self.num_codebooks if self.num_codebooks else None,
        }
        if self.vocab_size:
            stats['vocab_size'] = self.vocab_size
        return stats
    
    def __call__(self, item_ids: Union[int, List[int]]) -> torch.Tensor:
        """便捷调用方法"""
        return self.tokenize(item_ids)


def main():
    """测试 SemanticIdTokenizer 的各项功能"""
    from collections import defaultdict
    
    print("=" * 80)
    print("SemanticIdTokenizer 功能测试")
    print("=" * 80)

    # 初始化 tokenizer
    print("\n[测试 1] 初始化 Tokenizer")
    mapping_path = '/Users/yyds/workspace/REC/llm4rec-pytorch-simple/ml-1m/multimodal_datasets/rqvae_results/semantic_id_mapping_text.pkl'
    tokenizer = SemanticIdTokenizer.from_pretrained(mapping_path)

    # 查看统计信息
    stats = tokenizer.get_stats()
    print(f"\n统计信息:")
    print(f"  - 物品总数: {stats['total_items']}")
    print(f"  - Codebook 数量: {stats['num_codebooks']}")
    print(f"  - 扩展语义 ID 长度: {stats['semantic_id_length']}")
    if 'vocab_size' in stats:
        print(f"  - Vocab 大小: {stats['vocab_size']}")
    
    # 测试 tokenize 和 decode
    print("\n" + "=" * 80)
    print("[测试 2] Tokenize 和 Decode 可逆性")
    print("=" * 80)
    
    test_item_ids = [1, 2, 10, 100, 500]  # 使用 int
    print(f"\n测试物品: {test_item_ids}")
    
    # Tokenize
    semantic_ids = tokenizer.tokenize(test_item_ids)
    print(f"\nTokenize 结果 (形状 {semantic_ids.shape}):")
    for i, item_id in enumerate(test_item_ids):
        print(f"  物品 {item_id}: {semantic_ids[i].tolist()}")
    
    # Decode
    decoded_item_ids = tokenizer.decode(semantic_ids)
    print(f"\nDecode 结果: {decoded_item_ids}")
    
    # 验证可逆性
    success = all(original == decoded for original, decoded in zip(test_item_ids, decoded_item_ids))
    print(f"\n可逆性测试: {'✓ 通过' if success else '✗ 失败'}")
    
    # 测试冲突处理
    print("\n" + "=" * 80)
    print("[测试 3] 冲突处理统计")
    print("=" * 80)
    
    # 统计冲突
    original_codes_to_items = defaultdict(list)
    for item_id, semantic_ids in tokenizer.item_id_to_semantic_ids.items():
        original_code = tuple(semantic_ids[:-1])  # 去掉消歧索引
        original_codes_to_items[original_code].append((item_id, semantic_ids))
    
    conflicts = {k: v for k, v in original_codes_to_items.items() if len(v) > 1}
    
    print(f"\n冲突统计:")
    print(f"  - 有冲突的原始 code 序列数: {len(conflicts)}")
    print(f"  - 涉及的物品总数: {sum(len(items) for items in conflicts.values())}")
    
    if conflicts:
        print(f"\n冲突示例（前 2 个）:")
        for i, (code, items) in enumerate(list(conflicts.items())[:2]):
            print(f"\n  原始 code: {code}")
            print(f"  冲突物品数: {len(items)}")
            for item_id, extended_semantic_id in items[:3]:
                print(f"    - 物品 {item_id}: {extended_semantic_id} (消歧索引: {extended_semantic_id[-1]})")
    
    # 测试生成模式（简化版：不使用前缀验证）
    print("\n" + "=" * 80)
    print("[测试 4] 生成模式（简化版）")
    print("=" * 80)

    test_item_id = 1  # 使用 int
    semantic_ids = tokenizer.tokenize([test_item_id])[0].tolist()

    print(f"\n测试物品: {test_item_id}")
    print(f"完整语义 ID: {semantic_ids}")
    print(f"\n注意：简化版移除了 Trie 树校验")
    print(f"生成时会尝试所有 token，无效序列将在 decode 时返回 None")
    
    # 测试批量处理
    print("\n" + "=" * 80)
    print("[测试 5] 批量处理")
    print("=" * 80)
    
    batch_item_ids = list(range(1, 21))  # [1, 2, ..., 20]
    print(f"\n批量处理 {len(batch_item_ids)} 个物品...")
    
    semantic_ids = tokenizer.tokenize(batch_item_ids)
    print(f"Tokenize 结果形状: {semantic_ids.shape}")
    
    decoded_item_ids = tokenizer.decode(semantic_ids)
    success = all(original == decoded for original, decoded in zip(batch_item_ids, decoded_item_ids))
    print(f"批量可逆性测试: {'✓ 通过' if success else '✗ 失败'}")
    
    print("\n" + "=" * 80)
    print("所有测试完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

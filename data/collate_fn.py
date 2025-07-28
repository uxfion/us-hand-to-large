import torch
from torch.utils.data.dataloader import default_collate

def mixed_dataset_collate_fn(batch):
    """
    自定义的collate函数，用于处理混合数据集的batch
    确保mode信息能正确传递给模型
    """
    # 分离出各个字段
    A_list = [item['A'] for item in batch]
    B_list = [item['B'] for item in batch]
    A_paths_list = [item['A_paths'] for item in batch]
    B_paths_list = [item['B_paths'] for item in batch]
    mode_list = [item['mode'] for item in batch]
    
    # 使用默认的collate函数处理张量
    A_batch = default_collate(A_list)
    B_batch = default_collate(B_list)
    
    # 路径和模式保持为列表
    return {
        'A': A_batch,
        'B': B_batch,
        'A_paths': A_paths_list,
        'B_paths': B_paths_list,
        'mode': mode_list  # 这里是列表，包含batch中每个样本的模式
    }

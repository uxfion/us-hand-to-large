"""
使用混合数据集训练时的DataLoader配置示例
"""

import torch
from torch.utils.data import DataLoader
from data.mixed_dataset import MixedDataset
from data.collate_fn import mixed_dataset_collate_fn

def create_mixed_dataloader(opt):
    """
    创建使用混合数据集的DataLoader
    """
    dataset = MixedDataset(opt)
    
    # 使用自定义的collate函数
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.num_threads),
        collate_fn=mixed_dataset_collate_fn,  # 关键：使用自定义collate函数
        drop_last=True if opt.isTrain else False
    )
    
    return dataloader

# 在train.py或其他训练脚本中的使用示例：
"""
from data.mixed_dataloader import create_mixed_dataloader

# 创建数据加载器
train_dataloader = create_mixed_dataloader(opt)

for i, data in enumerate(train_dataloader):
    model.set_input(data)  # 现在data['mode']是一个列表
    model.optimize_parameters()
"""

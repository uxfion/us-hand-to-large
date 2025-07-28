#!/usr/bin/env python3
"""
测试混合数据集的配对损失功能
"""

import torch
import numpy as np
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

def test_paired_loss():
    """测试配对损失是否正确计算"""
    
    # 设置测试选项
    opt = TrainOptions().parse()
    opt.dataset_mode = 'mixed'
    opt.model = 'vq_cycle_gan'
    opt.batch_size = 4
    opt.lambda_paired = 10.0
    opt.gpu_ids = [0] if torch.cuda.is_available() else []
    
    # 创建数据集和模型
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Model: {type(model).__name__}")
    
    # 测试几个batch
    total_batches = 0
    paired_batches = 0
    total_paired_samples = 0
    
    for i, data in enumerate(dataset):
        if i >= 10:  # 只测试前10个batch
            break
            
        total_batches += 1
        print(f"\n--- Batch {i+1} ---")
        
        # 检查输入数据
        print(f"Batch size: {data['A'].size(0)}")
        print(f"Modes: {data['mode']}")
        
        # 设置输入并前向传播
        model.set_input(data)
        
        # 检查配对数据信息
        if hasattr(model, 'paired_mask'):
            num_paired = model.paired_mask.sum().item()
            print(f"Paired samples in batch: {num_paired}/{len(data['mode'])}")
            print(f"Paired mask: {model.paired_mask.tolist()}")
            
            if num_paired > 0:
                paired_batches += 1
                total_paired_samples += num_paired
        
        # 计算损失
        model.forward()
        model.backward_G()
        
        # 检查配对损失
        if hasattr(model, 'loss_paired'):
            print(f"Paired loss: {model.loss_paired.item():.6f}")
        else:
            print("No paired loss found!")
    
    print(f"\n=== Summary ===")
    print(f"Total batches tested: {total_batches}")
    print(f"Batches with paired data: {paired_batches}")
    print(f"Total paired samples: {total_paired_samples}")
    print(f"Average paired samples per batch: {total_paired_samples / total_batches:.2f}")

def test_collate_function():
    """测试collate函数是否正确处理模式信息"""
    from data.collate_fn import mixed_dataset_collate_fn
    
    # 创建测试数据
    batch = [
        {
            'A': torch.randn(3, 256, 256),
            'B': torch.randn(3, 256, 256),
            'A_paths': 'path1.jpg',
            'B_paths': 'path1.jpg',
            'mode': 'unaligned'
        },
        {
            'A': torch.randn(3, 256, 256),
            'B': torch.randn(3, 256, 256),
            'A_paths': 'path2.jpg',
            'B_paths': 'path2.jpg',
            'mode': 'aligned'
        },
        {
            'A': torch.randn(3, 256, 256),
            'B': torch.randn(3, 256, 256),
            'A_paths': 'path3.jpg',
            'B_paths': 'path3.jpg',
            'mode': 'semi_paired'
        }
    ]
    
    # 测试collate函数
    result = mixed_dataset_collate_fn(batch)
    
    print("=== Collate Function Test ===")
    print(f"A shape: {result['A'].shape}")
    print(f"B shape: {result['B'].shape}")
    print(f"A_paths: {result['A_paths']}")
    print(f"B_paths: {result['B_paths']}")
    print(f"Modes: {result['mode']}")
    print(f"Mode types: {[type(m) for m in result['mode']]}")

if __name__ == '__main__':
    print("Testing collate function...")
    test_collate_function()
    
    print("\n" + "="*50)
    print("Testing paired loss...")
    try:
        test_paired_loss()
    except Exception as e:
        print(f"Error during paired loss test: {e}")
        import traceback
        traceback.print_exc()

#!/usr/bin/env python3
"""
图像裁剪程序
使用 resize_and_crop 方式处理图像：先 resize 到 286x286，然后 crop 到 256x256
确保 HR 和 LR 图像使用相同的裁剪参数
"""

import os
import sys
import argparse
from PIL import Image
import random
import numpy as np

# 添加当前目录到 Python 路径，以便导入 data.base_dataset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.base_dataset import get_params, __crop
import torchvision.transforms as transforms


# def __transforms2pil_resize(method):
#     """将 torchvision 的插值方法转换为 PIL 的插值方法"""
#     mapper = {
#         transforms.InterpolationMode.BILINEAR: Image.BILINEAR,
#         transforms.InterpolationMode.BICUBIC: Image.BICUBIC,
#         transforms.InterpolationMode.NEAREST: Image.NEAREST,
#         transforms.InterpolationMode.LANCZOS: Image.LANCZOS,
#     }
#     return mapper[method]


def resize_and_crop_image(img_path, load_size, crop_size, crop_pos=None):
    """
    对图像进行 resize_and_crop 操作
    
    Args:
        img_path: 图像路径
        load_size: resize 后的尺寸
        crop_size: crop 后的尺寸  
        crop_pos: 裁剪位置 (x, y)，如果为 None 则随机裁剪
        
    Returns:
        processed_img: 处理后的 PIL 图像
        crop_pos: 实际使用的裁剪位置
    """
    # 打开图像
    img = Image.open(img_path).convert('RGB')
    
    # Resize 到 load_size x load_size
    method = Image.LANCZOS
    img_resized = img.resize((load_size, load_size), method)
    
    # 确定裁剪位置
    if crop_pos is None:
        # 随机选择裁剪位置
        max_offset = load_size - crop_size
        x = random.randint(0, max_offset) if max_offset > 0 else 0
        y = random.randint(0, max_offset) if max_offset > 0 else 0
        crop_pos = (x, y)
    
    # 执行裁剪
    x, y = crop_pos
    img_cropped = img_resized.crop((x, y, x + crop_size, y + crop_size))
    
    return img_cropped, crop_pos


def process_paired_images(hr_dir, lr_dir, hr_output_dir, lr_output_dir, load_size=286, crop_size=256):
    """
    处理成对的 HR 和 LR 图像，确保使用相同的裁剪参数
    
    Args:
        hr_dir: HR 图像输入目录
        lr_dir: LR 图像输入目录  
        hr_output_dir: HR 图像输出目录
        lr_output_dir: LR 图像输出目录
        load_size: resize 尺寸
        crop_size: crop 尺寸
    """
    # 创建输出目录
    os.makedirs(hr_output_dir, exist_ok=True)
    os.makedirs(lr_output_dir, exist_ok=True)
    
    # 获取 HR 图像列表
    hr_files = [f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    hr_files.sort()
    
    processed_count = 0
    
    for hr_file in hr_files:
        # 构造对应的 LR 文件名
        lr_file = hr_file.replace('_HR_', '_LR_')
        
        hr_path = os.path.join(hr_dir, hr_file)
        lr_path = os.path.join(lr_dir, lr_file)
        
        # 检查对应的 LR 文件是否存在
        if not os.path.exists(lr_path):
            print(f"警告: 找不到对应的 LR 文件: {lr_path}")
            continue
            
        try:
            # 处理 HR 图像（随机裁剪）
            hr_img, crop_pos = resize_and_crop_image(hr_path, load_size, crop_size)
            
            # 处理 LR 图像（使用相同的裁剪位置）  
            lr_img, _ = resize_and_crop_image(lr_path, load_size, crop_size, crop_pos)
            
            # 保存处理后的图像
            hr_output_path = os.path.join(hr_output_dir, hr_file)
            lr_output_path = os.path.join(lr_output_dir, lr_file)
            
            hr_img.save(hr_output_path)
            lr_img.save(lr_output_path)
            
            processed_count += 1
            print(f"已处理: {hr_file} -> crop_pos: {crop_pos}")
            
        except Exception as e:
            print(f"处理 {hr_file} 时出错: {str(e)}")
            continue
    
    print(f"\n处理完成！共处理了 {processed_count} 对图像")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='图像裁剪程序')
    parser.add_argument('--hr_dir', type=str, 
                       default='/root/exp/us-hand-to-large/datasets/xijing_split/test/test_semi_paired_HR',
                       help='HR 图像输入目录')
    parser.add_argument('--lr_dir', type=str,
                       default='/root/exp/us-hand-to-large/datasets/xijing_split/test/test_semi_paired_LR', 
                       help='LR 图像输入目录')
    parser.add_argument('--hr_output_dir', type=str,
                       default='/root/exp/us-hand-to-large/datasets/xijing_split/test/test_semi_paired_HR_crop',
                       help='HR 图像输出目录')
    parser.add_argument('--lr_output_dir', type=str,
                       default='/root/exp/us-hand-to-large/datasets/xijing_split/test/test_semi_paired_LR_crop',
                       help='LR 图像输出目录')
    parser.add_argument('--load_size', type=int, default=286, help='resize 尺寸')
    parser.add_argument('--crop_size', type=int, default=256, help='crop 尺寸')
    
    args = parser.parse_args()
    
    print("=== 图像裁剪程序 ===")
    print(f"HR 输入目录: {args.hr_dir}")
    print(f"LR 输入目录: {args.lr_dir}")
    print(f"HR 输出目录: {args.hr_output_dir}")
    print(f"LR 输出目录: {args.lr_output_dir}")
    print(f"Load size: {args.load_size}")
    print(f"Crop size: {args.crop_size}")
    print(f"预处理方式: resize_and_crop")
    print()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.hr_dir):
        print(f"错误: HR 输入目录不存在: {args.hr_dir}")
        return
        
    if not os.path.exists(args.lr_dir):
        print(f"错误: LR 输入目录不存在: {args.lr_dir}")
        return
    
    # 设置随机种子以便复现
    random.seed(42)
    np.random.seed(42)
    
    # 开始处理
    process_paired_images(
        args.hr_dir, 
        args.lr_dir,
        args.hr_output_dir, 
        args.lr_output_dir,
        args.load_size,
        args.crop_size
    )


if __name__ == '__main__':
    main()
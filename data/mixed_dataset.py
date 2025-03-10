import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random

class MixedDataset(BaseDataset):
    """混合数据集类，随机混合对齐和非对齐样本。
    
    在每个batch中同时包含：
    - 非对齐样本：从trainA和trainB目录加载
    - 对齐样本：从trainB加载并创建降质版本作为trainA
    """
    
    def __init__(self, opt):
        """初始化数据集类。"""
        BaseDataset.__init__(self, opt)
        # 设置目录并加载路径
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        # 设置变换
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        
        # 设置降质参数
        self.downscale_factor = opt.downscale_factor if hasattr(opt, 'downscale_factor') else 6
        self.interpolation = opt.interpolation if hasattr(opt, 'interpolation') else Image.BICUBIC
        
        # 创建混合索引映射
        self.create_mixed_indices()
        
    def create_mixed_indices(self):
        """创建混合的索引映射。
        
        生成两个数组：
        1. 对应于非对齐模式的索引数组
        2. 对应于对齐模式的索引数组
        
        然后将它们洗牌并合并，创建混合索引映射。
        """
        # 创建非对齐索引 (0 到 self.A_size-1)
        unaligned_indices = list(range(self.A_size))
        
        # 创建对齐索引 (0 到 self.B_size-1)，使用偏移表示它们是对齐模式
        aligned_indices = [i + self.A_size for i in range(self.B_size)]
        
        # 合并两个索引数组
        all_indices = unaligned_indices + aligned_indices
        
        # 打乱索引顺序
        random.shuffle(all_indices)
        
        # 存储混合索引映射
        self.mixed_indices = all_indices
        
    def __getitem__(self, index):
        """返回数据点及其元数据。
        
        使用混合索引映射确定该索引对应的实际模式和图像。
        """
        # 获取混合索引
        mixed_idx = self.mixed_indices[index]
        
        # 确定模式
        is_aligned = mixed_idx >= self.A_size
        
        if not is_aligned:
            # 非对齐模式：从A和B分别获取图像
            A_path = self.A_paths[mixed_idx]  # 直接使用混合索引（对应A_paths中的位置）
            if self.opt.serial_batches:
                index_B = mixed_idx % self.B_size
            else:
                index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')
        else:
            # 对齐模式：从B获取图像，创建降质版本作为A
            # 调整索引以访问B中的图像
            adjusted_index = (mixed_idx - self.A_size) % self.B_size
            B_path = self.B_paths[adjusted_index]
            B_img = Image.open(B_path).convert('RGB')
            A_img = self.degrade_image(B_img)
            # 为生成的A图像创建虚拟路径
            A_path = f"degraded_{B_path}"
        
        # 应用变换
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        # print(f"------AB path: {A_path}, {B_path}")
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'is_aligned': is_aligned}
        
    def __len__(self):
        """返回数据集中图像的总数（非对齐 + 对齐）。"""
        return len(self.mixed_indices)  # 总共 self.A_size + self.B_size
        
    def degrade_image(self, img):
        """创建输入图像的降质版本。"""
        # 获取原始尺寸
        w, h = img.size
        
        # 下采样
        w_low = w // self.downscale_factor
        h_low = h // self.downscale_factor
        img_low = img.resize((w_low, h_low), self.interpolation)
        
        # 上采样回原始尺寸
        degraded_img = img_low.resize((w, h), self.interpolation)
        
        return degraded_img
        
    def reset(self):
        """在每个epoch开始时重新洗牌索引。"""
        self.create_mixed_indices()
        print("in mixed_dataset.py, reset method called")
import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random

class MixedDataset(BaseDataset):
    """混合数据集类，根据指定权重混合非对齐、对齐和半对齐样本。
    
    包含三种训练模式与权重：
    - 非对齐样本 (权重=1)：从trainA和trainB目录加载
    - 对齐样本 (权重=1)：从trainB加载并创建降质版本作为trainA
    - 半对齐样本 (权重=6)：从semi_paired_LR和semi_paired_HR目录加载配对图像
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
        print(f"找到{self.A_size}张A图像")
        self.B_size = len(self.B_paths)
        print(f"找到{self.B_size}张B图像")
        
        # 设置半对齐数据路径
        self.dir_semi_LR = os.path.join(opt.dataroot, 'semi_paired_LR')
        self.dir_semi_HR = os.path.join(opt.dataroot, 'semi_paired_HR')
        
        # 检查目录是否存在
        self.has_semi_paired = os.path.exists(self.dir_semi_LR) and os.path.exists(self.dir_semi_HR)
        
        if self.has_semi_paired:
            self.semi_LR_paths = sorted(make_dataset(self.dir_semi_LR, opt.max_dataset_size))
            self.semi_HR_paths = sorted(make_dataset(self.dir_semi_HR, opt.max_dataset_size))
            self.semi_size = min(len(self.semi_LR_paths), len(self.semi_HR_paths))
            
            if self.semi_size > 0:
                print(f"找到{self.semi_size}对半对齐数据")
        else:
            self.semi_size = 0
            print("未找到半对齐数据目录")
        
        # 设置变换
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        
        # 设置降质参数
        self.downscale_factor = opt.downscale_factor if hasattr(opt, 'downscale_factor') else 6
        self.interpolation = opt.interpolation if hasattr(opt, 'interpolation') else Image.BICUBIC
        
        # 设置各模式权重
        self.unaligned_weight = 1
        self.aligned_weight = 1
        self.semi_paired_weight = 10
        
        # 计算基于权重的样本数量
        self.calculate_weighted_samples()
        
        # 创建基于权重的混合索引映射
        self.create_weighted_indices()
        
    def calculate_weighted_samples(self):
        """计算基于权重的各模式样本数量"""
        # 非配对模式使用较大的集合大小（通常是trainB）
        self.unaligned_samples = self.B_size  # 使用B_size而不是A_size
        
        # 配对模式使用B集合的大小
        self.aligned_samples = self.B_size
        
        # 半配对模式根据权重计算样本数
        self.semi_paired_samples = self.semi_size * self.semi_paired_weight
        
        # 计算总样本数
        self.total_samples = self.unaligned_samples + self.aligned_samples + self.semi_paired_samples
        
        print(f"基于权重的样本数: 非配对={self.unaligned_samples}, 配对={self.aligned_samples}, 半配对={self.semi_paired_samples}, 总计={self.total_samples}")
        
    def create_weighted_indices(self):
        """创建基于权重的混合索引映射，包含三种模式。"""
        # 创建非对齐索引，数量为unaligned_samples
        unaligned_indices = []
        for i in range(self.unaligned_samples):
            # 使用模运算确保索引在有效范围内
            unaligned_indices.append(('unaligned', i % self.A_size, i % self.B_size))
        
        # 创建对齐索引，数量为aligned_samples
        aligned_indices = []
        for i in range(self.aligned_samples):
            aligned_indices.append(('aligned', i % self.B_size))
        
        # 创建半对齐索引，数量为semi_paired_samples
        semi_paired_indices = []
        if self.semi_size > 0:
            for i in range(self.semi_paired_samples):
                semi_paired_indices.append(('semi_paired', i % self.semi_size))
        
        # 合并三个索引数组
        all_indices = unaligned_indices + aligned_indices + semi_paired_indices
        
        # 打乱索引顺序
        random.shuffle(all_indices)
        
        # 存储加权混合索引映射
        self.mixed_indices = all_indices
        
    def __getitem__(self, index):
        """返回数据点及其元数据。"""
        # 获取混合索引，现在是一个元组，包含模式和实际索引
        index_info = self.mixed_indices[index]
        mode = index_info[0]
        
        if mode == 'unaligned':
            # 非对齐模式：从A和B分别获取图像
            a_idx, b_idx = index_info[1], index_info[2]
            A_path = self.A_paths[a_idx]
            
            if self.opt.serial_batches:
                B_path = self.B_paths[b_idx]
            else:
                # 如果不使用serial_batches，随机选择B
                B_path = self.B_paths[random.randint(0, self.B_size - 1)]
                
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')
            
        elif mode == 'aligned':
            # 对齐模式：从B获取图像，创建降质版本作为A
            b_idx = index_info[1]
            B_path = self.B_paths[b_idx]
            B_img = Image.open(B_path).convert('RGB')
            A_img = self.degrade_image(B_img)
            A_path = f"degraded_{B_path}"
            
        else:  # mode == 'semi_paired'
            # 半对齐模式：从半对齐目录加载配对图像
            pair_idx = index_info[1]
            A_path = self.semi_LR_paths[pair_idx]
            B_path = self.semi_HR_paths[pair_idx]
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')
        
        # 应用变换
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        # print(f"mode: {mode}, A_path: {A_path}, B_path: {B_path}")
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'mode': mode}
        
    def __len__(self):
        """返回数据集中图像的总数（加权后的总样本数）。"""
        return len(self.mixed_indices)
        
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
        self.create_weighted_indices()
        print(f"混合数据集重置：总样本数={self.total_samples} (非配对={self.unaligned_samples}, 配对={self.aligned_samples}, 半配对={self.semi_paired_samples})")
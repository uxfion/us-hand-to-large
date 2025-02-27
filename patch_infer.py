import torch
from torchvision import transforms
from models import create_model
from options.test_options import TestOptions
import numpy as np
from PIL import Image

import os
from tqdm import tqdm
import sys


device='cuda:0' if torch.cuda.is_available() else 'cpu'
def load_cyclegan_model():
    # opt = TestOptions().parse()
    # opt.num_threads = 0
    # opt.batch_size = 1
    # opt.serial_batches = True
    # opt.no_flip = True
    # opt.display_id = -1
    #
    # opt.dataroot = "./dataset/all/test"
    # opt.name = "1.raw_rm_arrow"
    # opt.gpu_ids = [0]
    # opt.model = "test"
    # opt.no_dropout = True
    # opt.preprocess = "none"

    # 手动注入命令行参数
    if '--dataroot' not in sys.argv:
        sys.argv += ['--dataroot', './dataset/all/test']
    if '--name' not in sys.argv:
        sys.argv += ['--name', '1.raw_rm_arrow']
    if '--gpu_ids' not in sys.argv:
        sys.argv += ['--gpu_ids', '0']
    if '--model' not in sys.argv:
        sys.argv += ['--model', 'test']
    if '--no_dropout' not in sys.argv:   # 添加这个参数
        sys.argv += ['--no_dropout']
    if '--preprocess' not in sys.argv:
        sys.argv += ['--preprocess', 'none']
    if '--eval' not in sys.argv:         # 添加评估模式参数
        sys.argv += ['--eval']

    # 解析参数
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    # Load the model
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    return model

def create_gaussian_weight_matrix(patch_size):
    """创建高斯权重矩阵用于patch融合
    
    Args:
        patch_size (int): 图像块的大小
        
    Returns:
        torch.Tensor: 高斯权重矩阵
    """
    import torch
    center = patch_size / 2
    x = torch.arange(patch_size)
    y = torch.arange(patch_size)
    x, y = torch.meshgrid(x, y)
    
    # 计算到中心的距离
    gx = torch.exp(-(x - center)**2 / (2 * (patch_size/4)**2))
    gy = torch.exp(-(y - center)**2 / (2 * (patch_size/4)**2))
    g = gx * gy
    
    return g.to(device)

def extract_patches(image, patch_size=256, overlap=32):
    """从图像中提取重叠的patches
    
    Args:
        image (PIL.Image): 输入图像
        patch_size (int): patch的大小
        overlap (int): 重叠的像素数
        
    Returns:
        list: patches列表
        list: 位置信息列表 (x, y)
    """
    width, height = image.size
    stride = patch_size - overlap
    
    patches = []
    positions = []
    
    for y in range(0, height-overlap, stride):
        for x in range(0, width-overlap, stride):
            # 调整最后一个patch的位置，确保覆盖边缘
            if x + patch_size > width:
                x = width - patch_size
            if y + patch_size > height:
                y = height - patch_size
                
            # 提取patch
            patch = image.crop((x, y, x+patch_size, y+patch_size))
            patches.append(patch)
            positions.append((x, y))
            
            # 如果已经处理到最右边，跳出内循环
            if x + patch_size == width:
                break
        # 如果已经处理到最底部，跳出外循环
        if y + patch_size == height:
            break
            
    return patches, positions

def cyclegan_patch_infer(model, image_raw, patch_size=256, overlap=32):
    """使用patch-based策略进行全画幅推理
    
    Args:
        model: CycleGAN模型
        image_raw (PIL.Image): 输入图像
        patch_size (int): patch大小
        overlap (int): 重叠像素数
        
    Returns:
        PIL.Image: 处理后的图像
    """
    import torch
    import numpy as np
    from torchvision import transforms
    from PIL import Image
    
    # 确保输入图像是RGB模式
    if image_raw.mode != 'RGB':
        image_raw = image_raw.convert('RGB')
    
    # 保存原始尺寸
    width, height = image_raw.size
    print(f"Processing image of size {width}x{height}")
    
    # 创建结果张量
    result = torch.zeros((3, height, width)).to(device)
    weight_accumulator = torch.zeros((height, width)).to(device)
    
    # 获取高斯权重矩阵
    gaussian_weights = create_gaussian_weight_matrix(patch_size)
    
    # 提取patches
    patches, positions = extract_patches(image_raw, patch_size, overlap)
    print(f"Extracted {len(patches)} patches")
    
    # 准备转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 处理每个patch
    for idx, (patch, (x, y)) in enumerate(zip(patches, positions)):
        # 转换patch
        patch_tensor = transform(patch).unsqueeze(0).to(device)
        
        # 推理
        with torch.no_grad():
            fake_patch, _ = model.netG(patch_tensor)
        
        # 反归一化
        fake_patch = (fake_patch.squeeze(0) + 1) / 2.0
        
        # 应用高斯权重
        for c in range(3):  # 对每个通道
            result[c, y:y+patch_size, x:x+patch_size] += fake_patch[c] * gaussian_weights
        weight_accumulator[y:y+patch_size, x:x+patch_size] += gaussian_weights
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(patches)} patches")
    
    # 标准化结果
    final_result = result / weight_accumulator.unsqueeze(0).clamp(min=1e-8)
    
    # 转换回PIL图像
    final_result = torch.clamp(final_result, 0, 1)
    final_result = transforms.ToPILImage()(final_result.cpu())
    
    return final_result

def test_patch_inference():
    """测试函数
    """
    model = load_cyclegan_model()
    input_folder = './datasets/all/test/'
    output_folder = './results/test_patch'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    files = [f for f in os.listdir(input_folder) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
             
    for filename in tqdm(files, desc="Processing Images"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        image_raw = Image.open(input_path).convert("RGB")
        image_output = cyclegan_patch_infer(model, image_raw, 
                                          patch_size=256, overlap=32)
        image_output.save(output_path)

if __name__ == '__main__':
    test_patch_inference()
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
# device='cpu'

def load_cyclegan_model():
    # 手动注入命令行参数
    if '--dataroot' not in sys.argv:
        sys.argv += ['--dataroot', './dataset/all/test']
    if '--name' not in sys.argv:
        sys.argv += ['--name', '1.raw_rm_arrow_a401']
    if '--gpu_ids' not in sys.argv:
        sys.argv += ['--gpu_ids', '0']
    if '--model' not in sys.argv:
        sys.argv += ['--model', 'test']
    if '--no_dropout' not in sys.argv:
        sys.argv += ['--no_dropout']
    if '--preprocess' not in sys.argv:
        sys.argv += ['--preprocess', 'none']
    if '--eval' not in sys.argv:
        sys.argv += ['--eval']
    # 添加灰度图参数 - 确保模型接受单通道输入
    if '--input_nc' not in sys.argv:
        sys.argv += ['--input_nc', '1']
    if '--output_nc' not in sys.argv:
        sys.argv += ['--output_nc', '1']

    # 解析参数
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    
    # 确保模型知道要处理灰度图
    opt.input_nc = 1
    opt.output_nc = 1

    # Load the model
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    return model

def pad_to_next_power_2(image, base=256, max_size=None):
    """填充图像到下一个base的倍数，保持图像内容不变"""
    w, h = image.size
    new_w = ((w - 1) // base + 1) * base
    new_h = ((h - 1) // base + 1) * base
    
    # 创建新画布
    padded = Image.new(image.mode, (new_w, new_h))
    padded.paste(image, (0, 0))

    # 如果设置了max_size且尺寸超过限制，进行缩放
    if max_size is not None and (new_w > max_size or new_h > max_size):
        # 计算缩放比例
        ratio = max_size / max(new_w, new_h)
        final_w = int(new_w * ratio)
        final_h = int(new_h * ratio)
        # 确保缩放后的尺寸也是base的倍数
        final_w = ((final_w - 1) // base + 1) * base
        final_h = ((final_h - 1) // base + 1) * base
        # 缩放图像
        padded = padded.resize((final_w, final_h), Image.Resampling.LANCZOS)
    
    # 返回填充后的图像和原始尺寸信息
    return padded, (w, h)

def cyclegan_grayscale_infer(model, image_raw):
    """
    灰度图像的CycleGAN推理函数
    
    Args:
        model: CycleGAN模型
        image_raw: PIL Image格式的输入图像
    
    Returns:
        PIL Image格式的输出灰度图像
    """
    # 确保输入图像是灰度模式
    if image_raw.mode != 'L':
        image_raw = image_raw.convert('L')
    
    # 填充图像到4的倍数
    padded_image, (orig_w, orig_h) = pad_to_next_power_2(image_raw, base=4)
    
    # 打印尺寸信息用于调试
    print(f"Original size: {image_raw.size}, Padded size: {padded_image.size}")
    
    # 灰度图标准化转换 (对于灰度图，均值和标准差是单个值)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 单通道灰度图规范化
    ])
    
    # 转换图像
    image = transform(padded_image)
    
    # 添加批次维度并移到正确设备
    image = image.unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        fake_image = model.netG(image)
        
    # 转回 CPU 并处理
    fake_image = fake_image.cpu().squeeze(0)
    fake_image = (fake_image + 1) / 2.0
    fake_image = torch.clamp(fake_image, 0, 1)
    fake_image = transforms.ToPILImage()(fake_image)
    
    # 裁剪回原始尺寸
    fake_image = fake_image.crop((0, 0, orig_w, orig_h))
    
    return fake_image

def cyclegan_drop_others_grayscale_infer(model, image_raw):
    """
    灰度图像VQ-ResNet全画幅推理函数
    
    Args:
        model: CycleGAN模型
        image_raw: PIL Image格式的输入图像
    
    Returns:
        PIL Image格式的输出灰度图像
    """
    # 保存原始尺寸
    original_size = image_raw.size
    
    # 转换为灰度图
    if image_raw.mode != 'L':
        image_raw = image_raw.convert('L')
    
    # 填充图像到4的倍数
    padded_image, (orig_w, orig_h) = pad_to_next_power_2(image_raw, base=4, max_size=256)
    # 打印尺寸信息用于调试
    print(f"Original size: {image_raw.size}, Padded size: {padded_image.size}")

    
    # 灰度图标准化转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 单通道灰度图规范化
    ])
    
    # 转换图像
    image = transform(padded_image)
    image = image.unsqueeze(0).to(device)
    
    # 推理 - 适应模型输出结构
    with torch.no_grad():
        try:
            # 尝试使用返回两个值的接口
            fake_image, _ = model.netG(image)
        except ValueError:
            # 如果模型只返回一个值，直接使用
            fake_image = model.netG(image)
    
    # 转换回PIL图像
    fake_image = (fake_image.cpu().squeeze(0) + 1) / 2.0
    fake_image = torch.clamp(fake_image, 0, 1)
    fake_image = transforms.ToPILImage()(fake_image)
    
    # 调整回原始尺寸
    fake_image = fake_image.resize(original_size, Image.Resampling.LANCZOS)

    return fake_image



if __name__ == '__main__':
    model = load_cyclegan_model()
    # input_folder = '/root/Lecter/cyclegan-exp/us-hand-to-large/datasets/split/test/test_semi_paired_LR'  # 输入文件夹路径
    # input_folder = '/root/Lecter/cyclegan-exp/us-hand-to-large/datasets/split/test/test_only_LR'
    # input_folder = '/root/Lecter/dcm-convert/t1090000101al_gauss_subsample-dir-resizeto512nearest'
    # input_folder = '/root/Lecter/dcm-convert/sort/origin/t1090000101al-dir_shrink2'
    # input_folder = "/root/exp/us-hand-to-large/datasets/xijing_split/test/test_semi_paired_LR"
    input_folder = "/root/exp/us-hand-to-large/datasets/xijing_split/test/test_unpaired_LR"
    # input_folder = "/root/exp/us-hand-to-large/datasets/zhang/origin/t1090000101al-dir_shrink4"  # 输入文件夹路径
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  # 获取图片文件列表
    # output_folder = f'./results/results_semi_paired/1111'  # 输出文件夹路径
    # output_folder = f'./results/xijing_test_epoch195_LR_SR_x4_max256'
    output_folder = f'./results/xijing_test_LR_SR_x4_max256_unpaired'
    # output_folder = "./results/zhang/t1090000101al-dir_shrink4_sr"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in tqdm(files, desc="Processing Grayscale Images"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # 打开图像并转换为灰度
        image_raw = Image.open(input_path).convert("L")
        
        # 使用修改后的灰度图推理函数
        # 选择一种灰度图推理方法
        # image_output = cyclegan_grayscale_infer(model, image_raw)
        image_output = cyclegan_drop_others_grayscale_infer(model, image_raw)
        
        # 保存结果
        image_output.save(output_path)

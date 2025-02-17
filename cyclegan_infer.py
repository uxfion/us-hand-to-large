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


# Function to perform inference
def cyclegan_raw_infer(model, image_raw):
    # image_raw = crop_to_divisible_by_four(image_raw)
    # Apply necessary transformations
    transform = transforms.Compose([
        # transforms.Lambda(lambda img: __make_power_2(img, base=4)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image_raw).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        fake_image = model.netG(image.to("cuda:0"))
        # fake_image = model.netG(image.to("cuda:1"))

    # Convert to PIL image and return
    fake_image = (fake_image.cpu().squeeze(0) + 1) / 2.0  # Denormalize
    fake_image = transforms.ToPILImage()(fake_image)

    # contrast_image = mapped(image_raw, fake_image)

    return fake_image

def pad_to_next_power_2(image, base=256):
    """填充图像到下一个base的倍数，保持图像内容不变"""
    w, h = image.size
    new_w = ((w - 1) // base + 1) * base
    new_h = ((h - 1) // base + 1) * base
    
    # 创建新画布
    padded = Image.new(image.mode, (new_w, new_h))
    # 使用反射填充
    padded.paste(image, (0, 0))
    
    # 返回填充后的图像和原始尺寸信息
    return padded, (w, h)

def cyclegan_unet_infer(model, image_raw):
    """
    UNet 推理函数，使用填充方案处理任意尺寸输入
    
    Args:
        model: CycleGAN 模型
        image_raw: PIL Image 格式的输入图像
    
    Returns:
        PIL Image 格式的输出图像
    """
    # 确保输入图像是 RGB 模式
    if image_raw.mode != 'RGB':
        image_raw = image_raw.convert('RGB')
    
    # 填充图像到256的倍数
    padded_image, (orig_w, orig_h) = pad_to_next_power_2(image_raw, base=256)
    
    # 打印尺寸信息用于调试
    print(f"Original size: {image_raw.size}, Padded size: {padded_image.size}")
    
    # 标准化转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 转换图像
    image = transform(padded_image)
    
    # 添加批次维度并移到正确设备
    image = image.unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        model.eval()  # 确保模型在评估模式
        fake_image = model.netG(image)
        
        # 转回 CPU 并处理
        fake_image = fake_image.cpu().squeeze(0)
        fake_image = (fake_image + 1) / 2.0
        fake_image = torch.clamp(fake_image, 0, 1)
        fake_image = transforms.ToPILImage()(fake_image)
        
        # 裁剪回原始尺寸
        fake_image = fake_image.crop((0, 0, orig_w, orig_h))
    
    return fake_image


def cyclegan_vq_infer(model, image_raw):
    # 保存原始尺寸
    original_size = image_raw.size
    
    # 确保输入图像尺寸是4的倍数
    def make_size_divisible_by_4(size):
        return tuple(s - (s % 4) for s in size)
    
    # 计算新的目标尺寸
    # 确保是4的倍数，且不超过256
    target_size = make_size_divisible_by_4((
        min(256, original_size[0] - (original_size[0] % 4)),
        min(256, original_size[1] - (original_size[1] % 4))
    ))
    
    # 转换为RGB并调整大小
    if image_raw.mode != 'RGB':
        image_raw = image_raw.convert('RGB')
    image_resized = image_raw.resize(target_size, Image.Resampling.BICUBIC)
    
    # 打印调试信息
    print(f"Original size: {original_size}")
    print(f"Resized to: {target_size}")
    
    # 标准化转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = transform(image_resized)
    image = image.unsqueeze(0)
    
    # 执行推理
    with torch.no_grad():
        fake_image, _ = model.netG(image.to("cuda:0"))
    
    # 转换回PIL图像
    fake_image = (fake_image.cpu().squeeze(0) + 1) / 2.0
    fake_image = transforms.ToPILImage()(fake_image)
    
    # 调整回原始尺寸
    fake_image = fake_image.resize(original_size, Image.Resampling.BICUBIC)
    
    return fake_image

def cyclegan_new_vq_infer(model, image_raw):
    """
    VQ-ResNet 全画幅推理函数
    
    Args:
        model: CycleGAN 模型
        image_raw: PIL Image 格式的输入图像
    
    Returns:
        PIL Image 格式的输出图像
    """
    # 保存原始尺寸
    original_size = image_raw.size
    
    # 确保输入图像尺寸是4的倍数
    def make_size_divisible_by_4(size):
        return tuple(s + (4 - (s % 4)) if s % 4 != 0 else s for s in size)
    
    # 计算需要的填充尺寸
    target_size = make_size_divisible_by_4(original_size)
    
    # 转换为RGB
    if image_raw.mode != 'RGB':
        image_raw = image_raw.convert('RGB')
    
    # 如果需要，进行填充
    if target_size != original_size:
        # 创建新画布并粘贴原图
        padded_image = Image.new(image_raw.mode, target_size)
        padded_image.paste(image_raw, (0, 0))
        print(f"Padded from {original_size} to {target_size}")
    else:
        padded_image = image_raw
    
    # 标准化转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 转换图像
    image = transform(padded_image)
    image = image.unsqueeze(0)
    
    # 确保模型在评估模式
    model.netG.eval()
    
    # 执行推理
    try:
        with torch.no_grad():
            # 确保数据在正确的设备上
            device = next(model.netG.parameters()).device
            image = image.to(device)
            
            # 打印调试信息
            print(f"Input tensor shape: {image.shape}")
            fake_image, _ = model.netG(image)
            print(f"Output tensor shape: {fake_image.shape}")
    except RuntimeError as e:
        print(f"Error during inference: {str(e)}")
        raise e
    
    # 转换回PIL图像
    fake_image = (fake_image.cpu().squeeze(0) + 1) / 2.0
    fake_image = torch.clamp(fake_image, 0, 1)  # 确保像素值在有效范围内
    fake_image = transforms.ToPILImage()(fake_image)
    
    # 如果进行了填充，裁剪回原始尺寸
    if target_size != original_size:
        fake_image = fake_image.crop((0, 0, original_size[0], original_size[1]))
    
    return fake_image

# 调整第二张图像img2的亮度和对比度，使其与第一张图像img1相似。
def mapped(img1, img2):
    # 将 img1 和 img2 分解为三个通道
    img1_r, img1_g, img1_b = img1.split()
    img2_r, img2_g, img2_b = img2.split()

    # 分别对这三个通道进行处理
    img2_r = mapped_single_channel(img1_r, img2_r)
    img2_g = mapped_single_channel(img1_g, img2_g)
    img2_b = mapped_single_channel(img1_b, img2_b)

    # 将处理后的三个通道合并为一张彩色图像
    img2 = Image.merge("RGB", (img2_r, img2_g, img2_b))

    return img2


def mapped_single_channel(img1, img2):
    img1_pixels = np.sort(np.array(img1).flatten())
    img2_pixels = np.sort(np.array(img2).flatten())

    img1_low = float(img1_pixels[int(len(img1_pixels) * 0.05)])
    img1_high = float(img1_pixels[int(len(img1_pixels) * 0.95)])
    img2_low = float(img2_pixels[int(len(img2_pixels) * 0.05)])
    img2_high = float(img2_pixels[int(len(img2_pixels) * 0.95)])

    img2_array = np.array(img2, dtype=float)
    scale_factor = ((img2_array - img2_low) / (img2_high - img2_low)) * (img1_high - img1_low) + img1_low
    scale_factor = np.clip(scale_factor, 0, 255, out=scale_factor)

    return Image.fromarray(scale_factor.astype(np.uint8))


if __name__ == '__main__':
    # model = load_cyclegan_model()
    # image_raw = Image.open("./datasets/xijing/low_quality/1-LR.jpg")
    # image_output = cyclegan_infer(model, image_raw, 0)
    # image_output.save("./datasets/xijing/fake/1-LR-0.jpg")

    model = load_cyclegan_model()
    input_folder = './datasets/all/test/'  # 输入文件夹路径
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  # 获取图片文件列表
    output_folder = f'./results/test'  # 输出文件夹路径
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in tqdm(files, desc="Processing Images"):  # 使用tqdm显示进度
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        image_raw = Image.open(input_path).convert("RGB")
        image_output = cyclegan_new_vq_infer(model, image_raw)
        image_output.save(output_path)

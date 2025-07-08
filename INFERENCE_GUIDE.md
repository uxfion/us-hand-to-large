# CycleGAN Inference Guide - 优化版本

## 主要改进

- ✅ **复用框架代码**: 使用 `base_dataset.py` 中的 `get_transform` 函数
- ✅ **简化参数**: 减少冗余参数，专注核心功能  
- ✅ **标准化处理**: 与训练流程保持一致
- ✅ **更好的兼容性**: 完全兼容现有的 CycleGAN 框架

## 使用方法

### 1. Flexible Mode (默认，推荐)
使用 `--preprocess_mode flexible`，自动进行 power-of-2 填充

```bash
# 你的常用命令升级版
python infer_new.py --name vq_dual_xijing_5090d --netG vq_dual \
  --input_dir ./test_images --output_dir ./results

# 单张图像推理
python infer_new.py --name vq_dual_xijing_5090d --netG vq_dual \
  --input_path ./image.jpg --output_dir ./results
```

### 2. Standard Mode (与训练完全一致)
使用 `--preprocess_mode standard`，标准的 resize + crop 流程

```bash
# 标准模式：286->256 (与大多数训练一致)
python infer_new.py --name vq_dual_xijing_5090d --netG vq_dual \
  --preprocess_mode standard --load_size 286 --crop_size 256 \
  --input_dir ./test_images

# 高分辨率模式：512->512
python infer_new.py --name vq_dual_xijing_5090d --netG vq_dual \
  --preprocess_mode standard --load_size 512 --crop_size 512 \
  --input_dir ./test_images
```

## 对比原版的优势

### 代码层面
- **复用性**: 使用框架已有的 `get_transform` 函数
- **维护性**: 减少重复代码，降低维护成本
- **一致性**: 与训练时的数据处理完全一致

### 功能层面  
- **更简洁**: 参数更少，使用更直观
- **更可靠**: 使用经过验证的框架函数
- **更标准**: 完全遵循 CycleGAN 的设计理念

## 常用示例

### 你的标准用法
```bash
# 灰度图推理 (替代原来的 cyclegan_infer_gray.py)
python infer_new.py --name vq_dual_xijing_5090d --netG vq_dual \
  --input_dir ./path/to/images --output_dir ./results --verbose
```

### RGB 图像
```bash
# 彩色图像推理
python infer_new.py --name rgb_model --netG resnet_9blocks \
  --input_nc 3 --output_nc 3 \
  --input_dir ./rgb_images --output_dir ./results
```

### 单张图像快速测试
```bash
python infer_new.py --name vq_dual_xijing_5090d --netG vq_dual \
  --input_path ./test.jpg --output_name ./result.png --verbose
```

## 核心参数说明

- `--preprocess_mode`: 选择处理模式
  - `flexible` (默认): 自动 power-of-2 填充，保持比例
  - `standard`: 标准 resize+crop，与训练一致

- `--load_size` / `--crop_size`: 仅在 standard 模式下使用
  - 默认 286/256，与大多数 CycleGAN 训练一致

## 建议

对于日常使用，直接用默认设置即可：
```bash
python infer_new.py --name your_model_name --netG your_network \
  --input_dir ./your_images
```

这样既简单又可靠！

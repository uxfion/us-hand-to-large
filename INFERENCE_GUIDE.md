# CycleGAN Inference Guide - 优化版本

## 主要改进

- ✅ **复用框架代码**: 使用 `base_dataset.py` 中的 `get_transform` 函数
- ✅ **简化参数**: 减少冗余参数，专注核心功能  
- ✅ **标准化处理**: 与训练流程保持一致
- ✅ **更好的兼容性**: 完全兼容现有的 CycleGAN 框架

## 使用方法

### 1. Flexible Mode (默认，推荐)
使用 `--preprocess_mode flexible`，自动进行 power-of-2 调整

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

### 3. Fixed Size Mode (新增，推荐用于大图)
使用 `--preprocess_mode fixed_size`，指定尺寸推理但恢复原图大小

```bash
# 512x512 推理但输出原图尺寸
python infer_new.py --name vq_dual_xijing_5090d --netG vq_dual \
  --preprocess_mode fixed_size --fixed_size 512 \
  --input_dir ./test_images

# 1024x1024 高质量推理
python infer_new.py --name vq_dual_xijing_5090d --netG vq_dual \
  --preprocess_mode fixed_size --fixed_size 1024 \
  --input_path ./high_res_image.jpg

# 256x256 快速推理（适合显存受限）
python infer_new.py --name vq_dual_xijing_5090d --netG vq_dual \
  --preprocess_mode fixed_size --fixed_size 256 \
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
  - `flexible` (默认): power-of-2 调整，适合一般使用
  - `standard`: 标准 resize+crop，与训练一致
  - `fixed_size` (新增): 指定尺寸推理，恢复原图大小

- `--load_size` / `--crop_size`: 仅在 standard 模式下使用
  - 默认 286/256，与大多数 CycleGAN 训练一致

- `--fixed_size`: 仅在 fixed_size 模式下使用
  - 默认 512，推理时的固定尺寸

## 三种模式对比

| 模式 | 输入 999×999 | 推理尺寸 | 输出 | 特点 |
|------|-------------|----------|------|------|
| **flexible** | 999×999 | 1000×1000 | 999×999 | power-of-2 调整 |
| **standard** | 999×999 | 256×256 | 256×256 | 固定尺寸，可能变形 |
| **fixed_size** | 999×999 | 512×512 | 999×999 | 保持比例，恢复原尺寸 |

## 使用建议

### 🎯 **日常使用推荐**
```bash
# Fixed Size 模式 - 最佳平衡
python infer_new.py --name vq_dual_xijing_5090d --netG vq_dual \
  --preprocess_mode fixed_size --fixed_size 512 \
  --input_dir ./your_images
```

### 🚀 **高质量推理**
```bash
# 大尺寸推理获得更好效果
python infer_new.py --name vq_dual_xijing_5090d --netG vq_dual \
  --preprocess_mode fixed_size --fixed_size 1024 \
  --input_dir ./your_images
```

### ⚡ **快速推理**
```bash
# 小尺寸推理节省显存和时间
python infer_new.py --name vq_dual_xijing_5090d --netG vq_dual \
  --preprocess_mode fixed_size --fixed_size 256 \
  --input_dir ./your_images
```

## Fixed Size 模式详细说明

### 工作原理

**Fixed Size 模式**解决了现有模式的问题：
- **Flexible 模式**：999→1000→999 的变换可能影响质量
- **Standard 模式**：固定尺寸输出，丢失原图比例信息

### 处理流程示例（999×999 输入）

```
输入图像: 999×999
    ↓
等比缩放: 计算 scale = min(512/999, 512/999) = 0.512
    ↓  
缩放图像: 999×999 → 511×511
    ↓
居中放置: 在 512×512 黑色画布上居中放置 511×511 图像
    ↓
模型推理: 512×512 → 512×512
    ↓
提取区域: 从 512×512 输出中提取中心 511×511 区域
    ↓
恢复尺寸: 511×511 → 999×999 (resize back)
    ↓
最终输出: 999×999
```

### 优势对比

| 特性 | Flexible | Standard | **Fixed Size** |
|------|----------|----------|----------------|
| 保持原尺寸 | ✅ | ❌ | ✅ |
| 保持宽高比 | ✅ | ❌ | ✅ |
| 可控推理尺寸 | ❌ | ✅ | ✅ |
| 显存可控 | ❌ | ✅ | ✅ |
| 避免往返损失 | ❌ | ✅ | ✅ |

### 适用场景

#### ✅ **推荐使用 Fixed Size**
- 处理各种尺寸的图像
- 需要控制推理时的显存使用
- 希望在质量和效率间平衡
- 处理超大尺寸图像（如4K图像用512推理）

#### ✅ **适合的 fixed_size 设置**
- `256`: 快速推理，适合批量处理
- `512`: 平衡模式，推荐日常使用  
- `1024`: 高质量模式，适合精细处理
- `2048`: 超高质量，需要大显存

### 实际效果对比

对于一张 1920×1080 的图像：

```bash
# Flexible: 1920×1080 → 1920×1080 (可能显存不足)
python infer_new.py --preprocess_mode flexible

# Standard: 1920×1080 → 256×256 (丢失分辨率)
python infer_new.py --preprocess_mode standard

# Fixed Size: 1920×1080 → 512×512 → 1920×1080 (最佳平衡)
python infer_new.py --preprocess_mode fixed_size --fixed_size 512
```

**Fixed Size 模式是大多数场景的最佳选择！**

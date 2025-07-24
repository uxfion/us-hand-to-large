import os
import torch
import pyiqa
from glob import glob
import pandas as pd
from datetime import datetime
import json
import numpy as np

class ImageQualityEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化图像质量评估器
        
        Args:
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = device
        
        # 1. 全参考指标 (Full Reference) - 需要参考图像
        self.fr_metrics = {
            # 'LPIPS': pyiqa.create_metric('lpips', device=device),   # 感知损失，越低越好！！！
            # 'DISTS': pyiqa.create_metric('dists', device=device),   # 深度图像结构和纹理相似性，越低越好
            ####
            # 'PSNR': pyiqa.create_metric('psnry', device=device),    # 灰度图PSNR，越高越好
            # 'SSIM': pyiqa.create_metric('ssim', device=device),     # 灰度图SSIM，越高越好
            # 'MS_SSIM': pyiqa.create_metric('ms_ssim', device=device), # 多尺度SSIM，越高越好
            # 'CW_SSIM': pyiqa.create_metric('cw_ssim', device=device), # 复小波结构相似性，越高越好！！！
            # 'FSIM': pyiqa.create_metric('fsim', device=device),     # 特征相似性，越高越好
            # 'VIF': pyiqa.create_metric('vif', device=device),         # 视觉信息保真度，越高越好
            
        }
        
        # 2. 无参考指标 (No Reference) - 不需要参考图像
        self.nr_metrics = {
            'NIQE': pyiqa.create_metric('niqe', device=device),     # 自然图像质量评估，越低越好
            'BRISQUE': pyiqa.create_metric('brisque', device=device), # 盲图像质量评估，越低越好
            'PI': pyiqa.create_metric('pi', device=device),         # 感知指数，越低越好
            ###
            'TOPIQ_NR': pyiqa.create_metric('topiq_nr', device=device), # TOPIQ无参考版本，越高越好
            'ARNIQA': pyiqa.create_metric('arniqa', device=device),  # ARNIQA，越高越好
            'CLIPIQA': pyiqa.create_metric('clipiqa', device=device), # CLIPIQA，越高越好
            # 'MANIQA': pyiqa.create_metric('maniqa', device=device),  # MANIQA，越高越好
        }
        
        # 3. 特殊指标 - 需要特殊输入方式
        self.special_metrics = {
            'FID': pyiqa.create_metric('fid', device=device),       # 需要两个文件夹作为输入，越低越好
            # 'IS': pyiqa.create_metric('is', device=device),        # Inception Score，越高越好
            # 'KID': pyiqa.create_metric('kid', device=device),      # Kernel Inception Distance，越低越好
        }

    def evaluate_folder(self, folder_path, folder_name, fr_ref_folder=None, special_ref_folder=None):
        """
        评估单个文件夹的图像质量
        
        Args:
            folder_path: 待评估图像文件夹路径
            folder_name: 文件夹名称
            fr_ref_folder: 全参考指标的参考图像文件夹路径
            special_ref_folder: 特殊指标的参考文件夹路径
        """
        print(f"\nProcessing folder: {folder_name}")
        print("-" * 50)
        
        # 获取所有图像文件
        img_paths = glob(os.path.join(folder_path, '*.*'))
        if not img_paths:
            print(f"Warning: No images found in {folder_path}")
            return {}, []
        
        # 初始化结果存储
        results = {}
        
        # 添加无参考指标
        for metric_name in self.nr_metrics.keys():
            results[metric_name] = []
        
        # 如果提供了参考文件夹，添加全参考指标
        if fr_ref_folder:
            for metric_name in self.fr_metrics.keys():
                results[metric_name] = []
        
        image_results = []
        
        # 评估每张图像
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            try:
                scores = {}
                
                # 1. 计算无参考指标
                for metric_name, metric in self.nr_metrics.items():
                    score = metric(img_path).item()
                    scores[metric_name] = score
                    results[metric_name].append(score)
                
                # 2. 计算全参考指标
                if fr_ref_folder:
                    # 根据命名规律找到对应的参考图像
                    if '_LR_' in img_name:
                        ref_img_name = img_name.replace('_LR_', '_HR_')
                    else:
                        ref_img_name = img_name
                    
                    ref_img_path = os.path.join(fr_ref_folder, ref_img_name)
                    if os.path.exists(ref_img_path):
                        for metric_name, metric in self.fr_metrics.items():
                            score = metric(img_path, ref_img_path).item()
                            scores[metric_name] = score
                            results[metric_name].append(score)
                    else:
                        print(f"Warning: Reference image not found for {img_name} -> {ref_img_name}")
                        for metric_name in self.fr_metrics.keys():
                            scores[metric_name] = None
                            results[metric_name].append(None)
                
                scores['image_name'] = img_name
                image_results.append(scores)
                
                # 打印当前图像的结果
                print(f"Processed {img_name}:")
                for metric_name, score in scores.items():
                    if metric_name != 'image_name' and score is not None:
                        print(f"  {metric_name}: {score:.4f}")
                print("-" * 30)
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
        
        # 计算平均值和标准差
        avg_results = {}
        std_results = {}
        for metric_name, scores in results.items():
            valid_scores = [s for s in scores if s is not None]
            if valid_scores:
                avg_results[metric_name] = np.mean(valid_scores)
                std_results[metric_name] = np.std(valid_scores, ddof=1) if len(valid_scores) > 1 else 0.0
                print(f"Average {metric_name}: {avg_results[metric_name]:.4f} ± {std_results[metric_name]:.4f}")
            else:
                avg_results[metric_name] = None
                std_results[metric_name] = None
        
        # 3. 计算特殊指标
        if special_ref_folder:
            for metric_name, metric in self.special_metrics.items():
                try:
                    if metric_name == 'FID':
                        print(f"Calculating FID between {folder_path} and {special_ref_folder}")
                        score = metric(folder_path, special_ref_folder).item()
                        avg_results[metric_name] = score
                        std_results[metric_name] = None  # FID是单个值，没有标准差
                        print(f"FID score: {score:.4f}")
                    # 可以在这里添加其他特殊指标的计算逻辑
                except Exception as e:
                    print(f"Error calculating {metric_name}: {str(e)}")
                    avg_results[metric_name] = None
                    std_results[metric_name] = None
        
        return avg_results, std_results, image_results

    def calculate_fid(self, test_folder, ref_folder):
        """
        计算FID分数（已弃用，请使用evaluate_folder中的特殊指标）
        
        Args:
            test_folder: 测试图像文件夹
            ref_folder: 参考图像文件夹
        """
        print("Warning: calculate_fid is deprecated. Use special_metrics in evaluate_folder instead.")
        try:
            print(f"Calculating FID between:")
            print(f"Test folder: {test_folder}")
            print(f"Reference folder: {ref_folder}")
            fid_score = self.special_metrics['FID'](test_folder, ref_folder)
            print(f"FID score: {fid_score.item():.4f}")
            return fid_score.item()
        except Exception as e:
            print(f"Error calculating FID: {str(e)}")
            return None

    def print_metrics_info(self):
        """打印所有指标的信息"""
        print("=" * 80)
        print("IMAGE QUALITY METRICS INFORMATION")
        print("=" * 80)
        
        print("\n1. FULL REFERENCE METRICS (需要参考图像)")
        print("-" * 50)
        fr_info = {
            'PSNR': 'Peak Signal-to-Noise Ratio (峰值信噪比) - 越高越好',
            'SSIM': 'Structural Similarity Index (结构相似性指数) - 越高越好',
            'LPIPS': 'Learned Perceptual Image Patch Similarity (感知相似性) - 越低越好',
            'DISTS': 'Deep Image Structure and Texture Similarity - 越低越好',
            'FSIM': 'Feature Similarity Index Measure - 越高越好',
            'MS_SSIM': 'Multi-Scale Structural Similarity Index - 越高越好',
            'CW_SSIM': '复小波结构相似 - 越高越好',
        }
        for metric_name in self.fr_metrics.keys():
            if metric_name in fr_info:
                print(f"  ✓ {metric_name}: {fr_info[metric_name]}")
        
        print("\n2. NO REFERENCE METRICS (无参考指标)")
        print("-" * 50)
        nr_info = {
            'NIQE': 'Natural Image Quality Evaluator (自然图像质量评估) - 越低越好',
            'BRISQUE': 'Blind/Referenceless Image Spatial Quality Evaluator - 越低越好',
            'PI': 'Perceptual Index (感知指数) - 越低越好',
            'TOPIQ': 'TOPIQ No-Reference version - 越高越好',
            'ARNIQA': 'Aesthetic and Realistic No-Reference Image Quality Assessment - 越高越好',
            'CLIPIQA': 'CLIP-based Image Quality Assessment - 越高越好',
            'MANIQA': 'Multi-dimension Attention Network for Image Quality Assessment - 越高越好',
        }
        for metric_name in self.nr_metrics.keys():
            if metric_name in nr_info:
                print(f"  ✓ {metric_name}: {nr_info[metric_name]}")
        
        print("\n3. SPECIAL METRICS (特殊指标)")
        print("-" * 50)
        special_info = {
            'FID': 'Fréchet Inception Distance (需要两个文件夹) - 越低越好',
            'IS': 'Inception Score (需要文件夹) - 越高越好',
            'KID': 'Kernel Inception Distance (需要两个文件夹) - 越低越好',
        }
        for metric_name in self.special_metrics.keys():
            if metric_name in special_info:
                print(f"  ✓ {metric_name}: {special_info[metric_name]}")
        
        print("\n" + "=" * 80)

    def get_active_metrics(self):
        """获取当前激活的指标列表"""
        active_metrics = {
            'FR_metrics': list(self.fr_metrics.keys()),
            'NR_metrics': list(self.nr_metrics.keys()),
            'Special_metrics': list(self.special_metrics.keys())
        }
        return active_metrics

def save_results(all_results, output_dir='results/index'):
    """保存评估结果"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # 保存总结结果为CSV（包含均值和标准差）
    summary_df = pd.DataFrame.from_dict(all_results['summary'], orient='index')
    summary_df.to_csv(os.path.join(output_dir, f'summary_{timestamp}.csv'))
    
    # 保存标准差结果为CSV
    std_df = pd.DataFrame.from_dict(all_results['std'], orient='index')
    std_df.to_csv(os.path.join(output_dir, f'std_{timestamp}.csv'))
    
    # 保存合并的结果（均值±标准差格式）
    combined_df = pd.DataFrame(index=summary_df.index)
    for col in summary_df.columns:
        combined_df[col] = summary_df[col].combine(std_df[col], 
            lambda mean, std: f"{mean:.4f} ± {std:.4f}" if pd.notna(mean) and pd.notna(std) and std > 0 
            else f"{mean:.4f}" if pd.notna(mean) else "N/A")
    combined_df.to_csv(os.path.join(output_dir, f'combined_{timestamp}.csv'))
    
    # 保存详细结果为JSON
    with open(os.path.join(output_dir, f'detailed_{timestamp}.json'), 'w') as f:
        json.dump(all_results['detailed'], f, indent=4)
    
    # 生成报告
    report_path = os.path.join(output_dir, f'report_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write("Image Quality Assessment Report\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # 写入所有指标的结果，包括无参考指标和FID
        for folder in all_results['summary'].keys():
            f.write(f"\nFolder: {folder}\n")
            f.write("-" * 60 + "\n")
            for metric in all_results['summary'][folder].keys():
                mean_val = all_results['summary'][folder][metric]
                std_val = all_results['std'][folder][metric]
                
                if mean_val is not None:
                    if std_val is not None and std_val > 0:
                        f.write(f"{metric}: {mean_val:.4f} ± {std_val:.4f}\n")
                    else:
                        f.write(f"{metric}: {mean_val:.4f}\n")
                else:
                    f.write(f"{metric}: N/A\n")
            f.write("\n")

    print(f"\nResults saved to {output_dir}:")
    print(f"- Summary CSV (means): summary_{timestamp}.csv")
    print(f"- Standard deviation CSV: std_{timestamp}.csv")
    print(f"- Combined CSV (mean ± std): combined_{timestamp}.csv")
    print(f"- Detailed JSON: detailed_{timestamp}.json")
    print(f"- Report: report_{timestamp}.txt")
    
    return output_dir

def main():
    # 初始化评估器
    evaluator = ImageQualityEvaluator()
    
    # 显示当前激活的指标信息
    evaluator.print_metrics_info()
    
    # 设置基础路径
    base_path = "/root/exp/us-hand-to-large"
    
    # 定义需要评估的文件夹
    # 半配对数据
    semi_paired_folders = {
        'input': os.path.join(base_path, 'datasets/xijing_split/test/test_semi_paired_LR_crop_gray'),
        'gt': os.path.join(base_path, 'datasets/xijing_split/test/test_semi_paired_HR_crop_gray'),

        'real-esrgan': '/root/exp/Real-ESRGAN/results/test_cropdata_train0_SR_gray',
        'vanilla_cyclegan': '/root/exp/pytorch-CycleGAN-and-pix2pix/results/infer_new/xijing_test_vanilla_cyclegan_cropdata_flexNoResize',
        'vqresnet': os.path.join(base_path, 'results/infer_new/xijing_test_vqresnet_AtoB_cropdata_flexNoResize'),
        'vqdualv0': os.path.join(base_path, 'results/infer_new/xijing_test_vqdualv0_AtoB_cropdata_flexNoResize'),
        'vqdualv1(ours)': os.path.join(base_path, 'results/infer_new/xijing_test_vqdualv1_AtoB_cropdata_flexNoResize'),
    }

    unpaired_folders = {
        'input': os.path.join(base_path, 'datasets/xijing_split/trainA_crop_gray'),
        'gt': os.path.join(base_path, 'datasets/xijing_split/trainB_crop_gray'),

        'real-esrgan': '/root/exp/Real-ESRGAN/results/xijing_trainACropGray_RealESRGAN',
        'vanilla_cyclegan': '/root/exp/pytorch-CycleGAN-and-pix2pix/results/infer_new/xijing_trainACropGray_vanillaCyclegan_flexNoResize',
        'vqresnet': os.path.join(base_path, 'results/infer_new/xijing_trainACropGray_vqresnet_flexNoResize'),
        'vqdualv0': os.path.join(base_path, 'results/infer_new/xijing_trainACropGray_vqdualv0_AtoB_flexNoResize'),
        'vqdualv1(ours)': os.path.join(base_path, 'results/infer_new/xijing_trainACropGray_vqdualv1_AtoB_flexNoResize'),
    }

    folders_to_evaluate = unpaired_folders

    # FID参考文件夹（高清Ground Truth图像）
    # fid_ref_folder = os.path.join(base_path, 'datasets/xijing_split/trainB_gray')
    fid_ref_folder = os.path.join(base_path, 'datasets/xijing_split/trainB_crop_gray')
    
    # 全参考指标的参考文件夹（Ground Truth图像）
    fr_ref_folder = os.path.join(base_path, 'datasets/xijing_split/test/test_semi_paired_HR_crop_gray')
    
    # 显示当前激活的指标
    active_metrics = evaluator.get_active_metrics()
    print(f"\nActive metrics: {active_metrics}")
    
    # 存储所有结果
    all_results = {
        'summary': {},
        'std': {},
        'detailed': {}
    }
    
    # 评估每个文件夹
    for folder_name, folder_path in folders_to_evaluate.items():
        print(f"\nEvaluating {folder_name}...")
        
        # 确定参考文件夹
        if folder_name == 'gt':
            # gt文件夹与自己比较，验证全参考指标是否正常
            current_fr_ref = folder_path
        else:
            # 其他文件夹与gt比较
            current_fr_ref = fr_ref_folder
        
        # 评估当前文件夹
        avg_results, std_results, detailed_results = evaluator.evaluate_folder(
            folder_path=folder_path,
            folder_name=folder_name,
            fr_ref_folder=current_fr_ref,
            special_ref_folder=fid_ref_folder  # 所有文件夹都使用同一个FID参考
        )
        
        # 保存结果
        all_results['summary'][folder_name] = avg_results
        all_results['std'][folder_name] = std_results
        all_results['detailed'][folder_name] = detailed_results
    
    # 保存结果
    output_dir = save_results(all_results)
    
    # 打印总结表格
    print("\nSummary of Results:")
    print("=" * 80)
    
    # 创建一个整洁的表格显示（均值±标准差）
    summary_df = pd.DataFrame.from_dict(all_results['summary'], orient='index')
    std_df = pd.DataFrame.from_dict(all_results['std'], orient='index')
    
    # 创建合并显示的DataFrame
    combined_display = pd.DataFrame(index=summary_df.index)
    for col in summary_df.columns:
        combined_display[col] = summary_df[col].combine(std_df[col], 
            lambda mean, std: f"{mean:.4f}±{std:.4f}" if pd.notna(mean) and pd.notna(std) and std > 0 
            else f"{mean:.4f}" if pd.notna(mean) else "N/A")
    
    print("\nResults (Mean ± Std):")
    print(combined_display)
    
    print("\nMeans only:")
    print(summary_df.round(4))
    
    print("\nStandard deviations:")
    print(std_df.round(4))

if __name__ == "__main__":
    main()
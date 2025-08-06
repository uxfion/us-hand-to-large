import os
import torch
from glob import glob
import pandas as pd
from datetime import datetime
import json
import numpy as np
from my_iqa import calculate_nr_metric, calculate_fr_metric, calculate_fid

class ImageQualityEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化图像质量评估器
        
        Args:
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = device
        
        # 定义要计算的指标
        self.nr_metrics = ['niqe', 'brisque','topiq_nr', 'arniqa', 'clipiqa', 'pi']  # , , 
        self.fr_metrics = ['lpips', 'ssim', 'psnr']  # 如需要可以添加: []
        self.calculate_fid = False

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
        
        mean_results = {}
        std_results = {}
        metric_directions = {}  # 记录指标方向
        
        # 1. 计算无参考指标
        for metric_name in self.nr_metrics:
            try:
                print(f"Computing {metric_name.upper()}...")
                result = calculate_nr_metric(
                    metric_name=metric_name,
                    img_paths=folder_path,
                    batch_size=128,
                    device=self.device,
                    return_details=True
                )
                mean_results[metric_name.upper()] = result['mean_score']
                std_results[metric_name.upper()] = result['std_score']
                metric_directions[metric_name.upper()] = result.get('lower_better', None)
                print(f"  {metric_name.upper()}: {result['mean_score']:.4f} ± {result['std_score']:.4f}")
            except Exception as e:
                print(f"  Error calculating {metric_name}: {str(e)}")
                mean_results[metric_name.upper()] = None
                std_results[metric_name.upper()] = None
                metric_directions[metric_name.upper()] = None
        
        # 2. 计算全参考指标
        if fr_ref_folder and self.fr_metrics:
            for metric_name in self.fr_metrics:
                try:
                    print(f"Computing {metric_name.upper()}...")
                    result = calculate_fr_metric(
                        metric_name=metric_name,
                        dist_paths=folder_path,
                        ref_paths=fr_ref_folder,
                        batch_size=32,
                        device=self.device,
                        return_details=True
                    )
                    mean_results[metric_name.upper()] = result['mean_score']
                    std_results[metric_name.upper()] = result['std_score']
                    metric_directions[metric_name.upper()] = result.get('lower_better', None)
                    print(f"  {metric_name.upper()}: {result['mean_score']:.4f} ± {result['std_score']:.4f}")
                except Exception as e:
                    print(f"  Error calculating {metric_name}: {str(e)}")
                    mean_results[metric_name.upper()] = None
                    std_results[metric_name.upper()] = None
                    metric_directions[metric_name.upper()] = None
        
        # 3. 计算FID
        if special_ref_folder and self.calculate_fid:
            try:
                print(f"Computing FID between {folder_path} and {special_ref_folder}")
                fid_result = calculate_fid(
                    dist_dir=folder_path,
                    ref_dir=special_ref_folder,
                    device=self.device
                )
                # FID总是返回单个值，需要用特殊方式获取方向信息
                from my_iqa import IQACalculator
                fid_calculator = IQACalculator('fid', device=self.device)
                
                mean_results['FID'] = fid_result
                std_results['FID'] = None  # FID没有标准差
                metric_directions['FID'] = fid_calculator.lower_better
                print(f"  FID: {fid_result:.4f}")
            except Exception as e:
                print(f"  Error calculating FID: {str(e)}")
                mean_results['FID'] = None
                std_results['FID'] = None
                metric_directions['FID'] = None
        
        return mean_results, std_results, metric_directions

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
            fid_score = calculate_fid(test_folder, ref_folder, device=self.device)
            print(f"FID score: {fid_score:.4f}")
            return fid_score
        except Exception as e:
            print(f"Error calculating FID: {str(e)}")
            return None

    def print_metrics_info(self):
        """打印所有指标的信息"""
        print("=" * 80)
        print("IMAGE QUALITY METRICS INFORMATION")
        print("=" * 80)
        
        print("\n1. NO REFERENCE METRICS (无参考指标)")
        print("-" * 50)
        for metric in self.nr_metrics:
            print(f"  ✓ {metric.upper()}")
        
        if self.fr_metrics:
            print("\n2. FULL REFERENCE METRICS (需要参考图像)")
            print("-" * 50)
            for metric in self.fr_metrics:
                print(f"  ✓ {metric.upper()}")
        
        if self.calculate_fid:
            print("\n3. SPECIAL METRICS (特殊指标)")
            print("-" * 50)
            print(f"  ✓ FID")
        
        print("\n" + "=" * 80)

    def get_active_metrics(self):
        """获取当前激活的指标列表"""
        active_metrics = {
            'NR_metrics': self.nr_metrics,
            'FR_metrics': self.fr_metrics,
            'FID': self.calculate_fid
        }
        return active_metrics

def save_results(all_mean_results, all_std_results, metric_directions, output_dir='results/index'):
    """保存评估结果"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # 创建带方向标注的列名
    def add_direction_to_columns(df, directions):
        new_columns = []
        for col in df.columns:
            if col in directions and directions[col] is not None:
                if directions[col]:  # lower_better = True
                    new_columns.append(f"{col}↓")
                else:  # lower_better = False
                    new_columns.append(f"{col}↑")
            else:
                new_columns.append(col)
        return new_columns

    # 保存均值结果
    mean_df = pd.DataFrame.from_dict(all_mean_results, orient='index')
    mean_df.columns = add_direction_to_columns(mean_df, metric_directions)
    mean_df.to_csv(os.path.join(output_dir, f'results_mean_{timestamp}.csv'))
    
    # 保存标准差结果
    std_df = pd.DataFrame.from_dict(all_std_results, orient='index')
    std_df.columns = add_direction_to_columns(std_df, metric_directions)
    std_df.to_csv(os.path.join(output_dir, f'results_std_{timestamp}.csv'))
    
    # 保存合并结果（均值±标准差）
    all_df = pd.DataFrame(index=mean_df.index)
    for i, col in enumerate(mean_df.columns):
        mean_col = mean_df.iloc[:, i]
        std_col = std_df.iloc[:, i]
        all_df[col] = mean_col.combine(std_col, 
            lambda mean, std: f"{mean:.4f}±{std:.4f}" if pd.notna(mean) and pd.notna(std) and std is not None
            else f"{mean:.4f}" if pd.notna(mean) else "N/A")
    all_df.to_csv(os.path.join(output_dir, f'results_all_{timestamp}.csv'))
    
    # 保存详细结果为JSON
    results_json = {
        'mean_results': all_mean_results,
        'std_results': all_std_results,
        'metric_directions': metric_directions,
        'timestamp': timestamp
    }
    with open(os.path.join(output_dir, f'results_{timestamp}.json'), 'w') as f:
        json.dump(results_json, f, indent=4)
    
    # 生成报告
    report_path = os.path.join(output_dir, f'report_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write("Image Quality Assessment Report\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Metric Directions:\n")
        f.write("-" * 30 + "\n")
        for metric, direction in metric_directions.items():
            if direction is not None:
                arrow = "↓ (lower better)" if direction else "↑ (higher better)"
                f.write(f"{metric}: {arrow}\n")
        f.write("\n")
        
        for folder, metrics in all_mean_results.items():
            f.write(f"\nFolder: {folder}\n")
            f.write("-" * 60 + "\n")
            for metric, mean_value in metrics.items():
                std_value = all_std_results[folder].get(metric)
                if mean_value is not None:
                    if std_value is not None:
                        f.write(f"{metric}: {mean_value:.4f} ± {std_value:.4f}\n")
                    else:
                        f.write(f"{metric}: {mean_value:.4f}\n")
                else:
                    f.write(f"{metric}: N/A\n")
            f.write("\n")

    print(f"\nResults saved to {output_dir}:")
    print(f"- Mean results CSV: results_mean_{timestamp}.csv")
    print(f"- Std results CSV: results_std_{timestamp}.csv")
    print(f"- Combined results CSV: results_all_{timestamp}.csv")
    print(f"- Results JSON: results_{timestamp}.json")
    print(f"- Report: report_{timestamp}.txt")
    
    return output_dir, all_df  # 返回合并表格用于显示

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

    folders_to_evaluate = semi_paired_folders

    # FID参考文件夹（高清Ground Truth图像）
    # fid_ref_folder = os.path.join(base_path, 'datasets/xijing_split/trainB_gray')
    fid_ref_folder = os.path.join(base_path, 'datasets/xijing_split/trainB_crop_gray')
    
    # 全参考指标的参考文件夹（Ground Truth图像）
    fr_ref_folder = os.path.join(base_path, 'datasets/xijing_split/test/test_semi_paired_HR_crop_gray')
    
    # 显示当前激活的指标
    active_metrics = evaluator.get_active_metrics()
    print(f"\nActive metrics: {active_metrics}")
    
    # 存储所有结果
    all_mean_results = {}
    all_std_results = {}
    metric_directions = {}
    
    # 评估每个文件夹
    for folder_name, folder_path in folders_to_evaluate.items():
        print(f"\nEvaluating {folder_name}...")
        
        # 确定参考文件夹
        current_fr_ref = folder_path if folder_name == 'gt' else fr_ref_folder
        
        # 评估当前文件夹
        mean_results, std_results, directions = evaluator.evaluate_folder(
            folder_path=folder_path,
            folder_name=folder_name,
            fr_ref_folder=current_fr_ref,
            special_ref_folder=fid_ref_folder
        )
        
        all_mean_results[folder_name] = mean_results
        all_std_results[folder_name] = std_results
        
        # 更新指标方向信息（所有文件夹的指标方向应该是一致的）
        if not metric_directions:
            metric_directions = directions
    
    # 保存结果
    output_dir, all_df = save_results(all_mean_results, all_std_results, metric_directions)
    
    # 打印总结表格
    print("\nSummary of Results:")
    print("=" * 80)
    print("\nResults (Mean ± Std):")
    print(all_df)

if __name__ == "__main__":
    main()
import os
import torch
import pyiqa
from glob import glob
import pandas as pd
from datetime import datetime
import json

class ImageQualityEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # 初始化无参考指标
        self.metrics = {
            'NIQE': pyiqa.create_metric('niqe', device=device),
            'BRISQUE': pyiqa.create_metric('brisque', device=device),
            'PI': pyiqa.create_metric('pi', device=device)
        }
        # 初始化FID指标
        self.fid_metric = pyiqa.create_metric('fid', device=device)
        self.device = device

    def evaluate_folder(self, folder_path, folder_name):
        """评估单个文件夹的图像质量"""
        print(f"\nProcessing folder: {folder_name}")
        print("-" * 50)
        
        # 获取所有图像文件
        img_paths = glob(os.path.join(folder_path, '*.*'))
        results = {metric_name: [] for metric_name in self.metrics.keys()}
        image_results = []
        
        # 评估每张图像
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            try:
                # 计算所有指标
                scores = {}
                for metric_name, metric in self.metrics.items():
                    score = metric(img_path).item()
                    scores[metric_name] = score
                    results[metric_name].append(score)
                
                scores['image_name'] = img_name
                image_results.append(scores)
                
                print(f"Processed {img_name}:")
                for metric_name, score in scores.items():
                    if metric_name != 'image_name':
                        print(f"{metric_name}: {score:.4f}")
                print("-" * 30)
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
        
        # 计算平均值
        avg_results = {
            metric: sum(scores)/len(scores) if scores else None
            for metric, scores in results.items()
        }
        
        return avg_results, image_results

    def calculate_fid(self, test_folder, ref_folder):
        """计算FID分数"""
        try:
            print(f"Calculating FID between:")
            print(f"Test folder: {test_folder}")
            print(f"Reference folder: {ref_folder}")
            fid_score = self.fid_metric(test_folder, ref_folder)
            print(f"FID score: {fid_score.item():.4f}")
            return fid_score.item()
        except Exception as e:
            print(f"Error calculating FID: {str(e)}")
            return None

def save_results(all_results, output_dir='evaluation_results'):
    """保存评估结果"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存总结结果为CSV
    summary_df = pd.DataFrame.from_dict(all_results['summary'], orient='index')
    summary_df.to_csv(os.path.join(output_dir, f'summary_{timestamp}.csv'))
    
    # 保存详细结果为JSON
    with open(os.path.join(output_dir, f'detailed_{timestamp}.json'), 'w') as f:
        json.dump(all_results['detailed'], f, indent=4)
    
    # 生成报告
    report_path = os.path.join(output_dir, f'report_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write("Image Quality Assessment Report\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        # 写入所有指标的结果，包括无参考指标和FID
        for folder, metrics in all_results['summary'].items():
            f.write(f"\nFolder: {folder}\n")
            f.write("-" * 40 + "\n")
            for metric, value in metrics.items():
                if value is not None:
                    f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")

    print(f"\nResults saved to {output_dir}:")
    print(f"- Summary CSV: summary_{timestamp}.csv")
    print(f"- Detailed JSON: detailed_{timestamp}.json")
    print(f"- Report: report_{timestamp}.txt")
    
    return output_dir

def main():
    # 设置基础路径
    base_path = "/root/Lecter/cyclegan-exp/us-hand-to-large"
    
    # 定义需要评估的文件夹
    folders_to_evaluate = {
        # 'original': os.path.join(base_path, 'datasets/all/test'),
        # 'resnet_9block': os.path.join(base_path, 'results/test_1.raw_rm_arrow_a401'),
        # 'unet256_resize': os.path.join(base_path, 'results/test_2.raw_unet256_resize_256_3090'),
        # 'unet256_pad': os.path.join(base_path, 'results/test_2.raw_unet256_pad_1024_3090'),
        # 'hybrid_restormer': os.path.join(base_path, 'results/test_xxx.hybrid_restormer_1'),
        'vq_resnet_resize': os.path.join(base_path, 'results/test_3.vq_resnet_resize_256_3090'),
        'vq_resnet_up_4x': os.path.join(base_path, 'results/test_3.vq_resnet_up_4x_3090'),
    }
    
    # FID参考文件夹（高清Ground Truth图像）
    fid_ref_folder = os.path.join(base_path, 'datasets/all/fid_ref')
    
    # 初始化评估器
    evaluator = ImageQualityEvaluator()
    
    # 存储所有结果
    all_results = {
        'summary': {},
        'detailed': {}
    }
    
    # 评估每个文件夹
    for folder_name, folder_path in folders_to_evaluate.items():
        print(f"\nEvaluating {folder_name}...")
        
        # 计算无参考指标
        avg_results, detailed_results = evaluator.evaluate_folder(folder_path, folder_name)
        
        # 对所有文件夹都计算FID
        fid_score = evaluator.calculate_fid(folder_path, fid_ref_folder)
        avg_results['FID'] = fid_score
        
        # 保存结果
        all_results['summary'][folder_name] = avg_results
        all_results['detailed'][folder_name] = detailed_results
    
    # 保存结果
    output_dir = save_results(all_results)
    
    # 打印总结表格
    print("\nSummary of Results:")
    print("=" * 50)
    
    # 创建一个整洁的表格显示
    summary_df = pd.DataFrame.from_dict(all_results['summary'], orient='index')
    print("\n", summary_df.round(4), "\n")

if __name__ == "__main__":
    main()
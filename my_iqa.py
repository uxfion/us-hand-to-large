import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import re
from tqdm import tqdm
import pyiqa
from typing import List, Union, Dict, Optional, Tuple
import numpy as np
from pathlib import Path


class IQACalculator:
    """
    Universal IQA Calculator supporting No Reference, Full Reference, and Special Reference metrics.
    
    Metric Types:
    - No Reference (NR): brisque, niqe, ilniqe, piqe, dbcnn, musiq, etc.
    - Full Reference (FR): psnr, ssim, lpips, dists, fsim, vif, etc.
    - Special Reference: fid, kid, etc. (require directory inputs)
    """
    
    # Define metric types
    NR_METRICS = {
        'brisque', 'niqe', 'ilniqe', 'piqe', 'dbcnn', 'musiq', 'hyperiqa',
        'maniqa', 'clipiqa', 'clipiqa+', 'tres', 'topiq_nr', 'arniqa',
        'cnniqa', 'wadiqam_nr', 'paq2piq', 'nima', 'pi', 'nrqm',
        'qualiclip', 'qualiclip+', 'liqe', 'qalign'
    }
    
    FR_METRICS = {
        'psnr', 'ssim', 'ms_ssim', 'lpips', 'dists', 'fsim', 'vif',
        'gmsd', 'nlpd', 'vsi', 'mad', 'topiq_fr', 'ahiq', 'pieapp',
        'wadiqam_fr', 'ckdn', 'cw_ssim', 'ssimc', 'psnry', 'stlpips'
    }
    
    SPECIAL_METRICS = {
        'fid', 'kid', 'fid_dinov2', 'sfid'
    }
    
    def __init__(self, metric_name: str, device: str = 'cuda', **metric_kwargs):
        """
        Initialize IQA Calculator.
        
        Args:
            metric_name: Name of the IQA metric
            device: Device to run the metric on ('cuda' or 'cpu')
            **metric_kwargs: Additional arguments for metric initialization
        """
        self.metric_name = metric_name.lower()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.metric_type = self._get_metric_type()
        
        # Initialize metric
        self.metric = pyiqa.create_metric(metric_name, device=self.device, **metric_kwargs)
        self.lower_better = getattr(self.metric, 'lower_better', None)
        
        # Image transform
        self.transform = transforms.ToTensor()
        
    def _get_metric_type(self) -> str:
        """Determine metric type based on metric name."""
        if self.metric_name in self.NR_METRICS:
            return 'NR'
        elif self.metric_name in self.FR_METRICS:
            return 'FR'
        elif self.metric_name in self.SPECIAL_METRICS:
            return 'SPECIAL'
        else:
            # Try to infer from metric attributes
            print(f"Warning: Unknown metric type for '{self.metric_name}'. Attempting to auto-detect...")
            return 'AUTO'
    
    def calculate_from_paths(
        self,
        dist_paths: Union[str, List[str]],
        ref_paths: Optional[Union[str, List[str]]] = None,
        batch_size: int = 32,
        return_details: bool = False
    ) -> Union[float, Dict[str, Union[float, List[float]]]]:
        """
        Calculate IQA scores from image paths or directories.
        
        Args:
            dist_paths: Path(s) to distorted images or directory
            ref_paths: Path(s) to reference images or directory (for FR metrics)
            batch_size: Batch size for processing
            return_details: If True, return detailed results
            
        Returns:
            Average score or detailed results dictionary
        """
        # Handle special metrics (FID, KID, etc.)
        if self.metric_type == 'SPECIAL':
            return self._calculate_special(dist_paths, ref_paths)
        
        # Convert paths to lists
        dist_list = self._prepare_paths(dist_paths)
        ref_list = self._prepare_paths(ref_paths) if ref_paths else None
        
        # Validate inputs
        if self.metric_type == 'FR' and (ref_list is None or len(dist_list) != len(ref_list)):
            raise ValueError(f"FR metric '{self.metric_name}' requires matching reference images")
        
        # Calculate scores
        if self.metric_type == 'NR' or (self.metric_type == 'AUTO' and ref_list is None):
            results = self._calculate_nr_batch(dist_list, batch_size)
        else:
            results = self._calculate_fr_batch(dist_list, ref_list, batch_size)
        
        if return_details:
            return results
        else:
            return results['mean_score']
    
    def calculate_from_tensors(
        self,
        dist_tensors: torch.Tensor,
        ref_tensors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate IQA scores from tensors.
        
        Args:
            dist_tensors: Distorted image tensors (N, 3, H, W), RGB, 0~1
            ref_tensors: Reference image tensors (N, 3, H, W), RGB, 0~1
            
        Returns:
            IQA scores tensor
        """
        if self.metric_type == 'SPECIAL':
            raise ValueError(f"Special metric '{self.metric_name}' requires directory inputs")
        
        # Move tensors to device
        dist_tensors = dist_tensors.to(self.device)
        if ref_tensors is not None:
            ref_tensors = ref_tensors.to(self.device)
        
        # Calculate scores
        with torch.no_grad():
            if self.metric_type == 'NR' or (self.metric_type == 'AUTO' and ref_tensors is None):
                scores = self.metric(dist_tensors)
            else:
                if ref_tensors is None:
                    raise ValueError(f"Metric '{self.metric_name}' requires reference tensors")
                scores = self.metric(dist_tensors, ref_tensors)
        
        return scores
    
    def _prepare_paths(self, paths: Union[str, List[str]]) -> List[str]:
        """Convert path input to list of image paths."""
        if isinstance(paths, str):
            if os.path.isdir(paths):
                # Get all images from directory
                return self._get_images_from_dir(paths)
            else:
                return [paths]
        elif isinstance(paths, list):
            # Expand directories in list
            expanded = []
            for path in paths:
                if os.path.isdir(path):
                    expanded.extend(self._get_images_from_dir(path))
                else:
                    expanded.append(path)
            return expanded
        else:
            raise ValueError(f"Invalid path type: {type(paths)}")
    
    def _get_images_from_dir(self, directory: str) -> List[str]:
        """Get all image paths from directory with natural sorting."""
        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
        img_paths = []
        
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in extensions):
                img_paths.append(os.path.join(directory, file))
        
        # Natural sort
        img_paths.sort(key=self._natural_sort_key)
        return img_paths
    
    @staticmethod
    def _natural_sort_key(path: str):
        """Natural sorting key function."""
        filename = os.path.basename(path)
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split('([0-9]+)', filename)]
    
    def _load_image_tensor(self, img_path: str) -> torch.Tensor:
        """Load image and convert to tensor."""
        try:
            img = Image.open(img_path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None
    
    def _calculate_nr_batch(
        self,
        img_paths: List[str],
        batch_size: int
    ) -> Dict[str, Union[float, List[float]]]:
        """Calculate NR metric scores in batches."""
        all_scores = []
        all_filenames = []
        failed_files = []
        
        print(f"Calculating {self.metric_name} scores for {len(img_paths)} images...")
        
        for i in tqdm(range(0, len(img_paths), batch_size), desc="Processing batches"):
            batch_paths = img_paths[i:i+batch_size]
            batch_tensors = []
            batch_filenames = []
            
            # Load batch
            for img_path in batch_paths:
                tensor = self._load_image_tensor(img_path)
                if tensor is not None:
                    batch_tensors.append(tensor)
                    batch_filenames.append(os.path.basename(img_path))
                else:
                    failed_files.append(img_path)
            
            if batch_tensors:
                # Stack and calculate
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                with torch.no_grad():
                    scores = self.metric(batch_tensor)
                    if isinstance(scores, torch.Tensor):
                        scores = scores.cpu().numpy()
                    
                    if isinstance(scores, np.ndarray):
                        all_scores.extend(scores.flatten().tolist())
                    else:
                        all_scores.append(float(scores))
                    
                    all_filenames.extend(batch_filenames)
        
        return self._compile_results(all_scores, all_filenames, failed_files)
    
    def _calculate_fr_batch(
        self,
        dist_paths: List[str],
        ref_paths: List[str],
        batch_size: int
    ) -> Dict[str, Union[float, List[float]]]:
        """Calculate FR metric scores in batches."""
        all_scores = []
        all_filenames = []
        failed_files = []
        
        print(f"Calculating {self.metric_name} scores for {len(dist_paths)} image pairs...")
        
        for i in tqdm(range(0, len(dist_paths), batch_size), desc="Processing batches"):
            batch_dist_paths = dist_paths[i:i+batch_size]
            batch_ref_paths = ref_paths[i:i+batch_size]
            
            batch_dist_tensors = []
            batch_ref_tensors = []
            batch_filenames = []
            
            # Load batch pairs
            for dist_path, ref_path in zip(batch_dist_paths, batch_ref_paths):
                dist_tensor = self._load_image_tensor(dist_path)
                ref_tensor = self._load_image_tensor(ref_path)
                
                if dist_tensor is not None and ref_tensor is not None:
                    batch_dist_tensors.append(dist_tensor)
                    batch_ref_tensors.append(ref_tensor)
                    batch_filenames.append(os.path.basename(dist_path))
                else:
                    failed_files.append((dist_path, ref_path))
            
            if batch_dist_tensors:
                # Stack and calculate
                dist_batch = torch.stack(batch_dist_tensors).to(self.device)
                ref_batch = torch.stack(batch_ref_tensors).to(self.device)
                
                with torch.no_grad():
                    scores = self.metric(dist_batch, ref_batch)
                    if isinstance(scores, torch.Tensor):
                        scores = scores.cpu().numpy()
                    
                    if isinstance(scores, np.ndarray):
                        all_scores.extend(scores.flatten().tolist())
                    else:
                        all_scores.append(float(scores))
                    
                    all_filenames.extend(batch_filenames)
        
        return self._compile_results(all_scores, all_filenames, failed_files)
    
    def _calculate_special(
        self,
        dist_path: str,
        ref_path: Optional[str]
    ) -> Dict[str, float]:
        """Calculate special metrics like FID."""
        if not os.path.isdir(dist_path):
            raise ValueError(f"Special metric '{self.metric_name}' requires directory input")
        
        if ref_path and not os.path.isdir(ref_path):
            raise ValueError(f"Special metric '{self.metric_name}' requires reference directory")
        
        print(f"Calculating {self.metric_name} between directories...")
        
        if ref_path:
            score = self.metric(dist_path, ref_path)
        else:
            # Some special metrics might support single directory
            score = self.metric(dist_path)
        
        return {
            'mean_score': float(score),
            'metric_name': self.metric_name,
            'lower_better': self.lower_better,
            'dist_path': dist_path,
            'ref_path': ref_path
        }
    
    def _compile_results(
        self,
        scores: List[float],
        filenames: List[str],
        failed_files: List
    ) -> Dict[str, Union[float, List[float]]]:
        """Compile results into structured dictionary."""
        if not scores:
            raise ValueError("No valid scores calculated")
        
        results = {
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'scores': scores,
            'filenames': filenames,
            'metric_name': self.metric_name,
            'lower_better': self.lower_better,
            'num_images': len(scores),
            'failed_files': failed_files
        }
        
        return results


# Convenience functions
def calculate_nr_metric(
    metric_name: str,
    img_paths: Union[str, List[str]],
    batch_size: int = 32,
    device: str = 'cuda',
    return_details: bool = False,
    **metric_kwargs
) -> Union[float, Dict]:
    """Calculate No Reference metric scores."""
    calculator = IQACalculator(metric_name, device, **metric_kwargs)
    return calculator.calculate_from_paths(img_paths, batch_size=batch_size, 
                                         return_details=return_details)


def calculate_fr_metric(
    metric_name: str,
    dist_paths: Union[str, List[str]],
    ref_paths: Union[str, List[str]],
    batch_size: int = 32,
    device: str = 'cuda',
    return_details: bool = False,
    **metric_kwargs
) -> Union[float, Dict]:
    """Calculate Full Reference metric scores."""
    calculator = IQACalculator(metric_name, device, **metric_kwargs)
    return calculator.calculate_from_paths(dist_paths, ref_paths, 
                                         batch_size=batch_size,
                                         return_details=return_details)


def calculate_fid(
    dist_dir: str,
    ref_dir: str,
    device: str = 'cuda',
    **fid_kwargs
) -> float:
    """Calculate FID score between two directories."""
    calculator = IQACalculator('fid', device, **fid_kwargs)
    results = calculator.calculate_from_paths(dist_dir, ref_dir)
    return results['mean_score'] if isinstance(results, dict) else results


# Example usage
if __name__ == "__main__":
    # Example 1: No Reference metric
    nr_score = calculate_nr_metric(
        'brisque',
        '/root/exp/us-hand-to-large/results/infer_new/xijing_trainACropGray_vqdualv0_AtoB_flexNoResize',
        batch_size=64,
        device='cuda'
    )
    print(f"NR Score: {nr_score}")
    
    # Example 2: Full Reference metric  
    fr_score = calculate_fr_metric(
        'lpips',
        '/root/exp/us-hand-to-large/results/infer_new/xijing_test_vqdualv1_AtoB_cropdata_flexNoResize',
        '/root/exp/us-hand-to-large/datasets/xijing_split/test/test_semi_paired_HR_crop_gray',
        batch_size=32,
        device='cuda'
    )
    print(f"FR Score: {fr_score}")
    
    # Example 3: FID score
    fid_score = calculate_fid(
        '/root/exp/us-hand-to-large/results/infer_new/xijing_trainACropGray_vqdualv0_AtoB_flexNoResize',
        '/root/exp/us-hand-to-large/datasets/xijing_split/trainB_crop_gray',
        device='cuda'
    )
    print(f"FID Score: {fid_score}")
    
    # Example 4: Using calculator class with tensors
    calculator = IQACalculator('ssim', device='cuda')
    
    # Prepare tensors (N, 3, H, W), RGB, 0~1
    dist_tensors = torch.rand(10, 3, 256, 256)
    ref_tensors = torch.rand(10, 3, 256, 256)
    
    scores = calculator.calculate_from_tensors(dist_tensors, ref_tensors)
    print(f"Tensor scores: {scores}")
    
    # Example 5: Get detailed results
    detailed_results = calculate_nr_metric(
        'musiq',
        ['img1.png', 'img2.png', 'img3.png'],
        return_details=True
    )
    print(f"Mean: {detailed_results['mean_score']:.4f}")
    print(f"Std: {detailed_results['std_score']:.4f}")
    print(f"Individual scores: {detailed_results['scores']}")
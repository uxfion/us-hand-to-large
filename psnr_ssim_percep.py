import os
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
import glob

import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.models.vgg import VGG16_Weights
import torchvision.transforms as transforms

blur = 8
original_dir = './datasets/all/test'
# results_base_dir = './results/xijing/9restormer'

# Find all test directories matching the pattern
# test_dirs = glob.glob(f'{results_base_dir}/test_degradation{blur}_sr*')
# test_dirs.sort()  # Sort to process in order
test_dirs = [
    "./results/test_1.raw_rm_arrow_a401/",
    # "./results/test_2.hybrid_restormer/"
]

print(f"Found {len(test_dirs)} test directories")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VGG normalization constants
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:23])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.to(device)

    def forward(self, input, target):
        # Normalize input and target
        input = (input - mean) / std
        target = (target - mean) / std
        
        input_features = self.feature_extractor(input)
        target_features = self.feature_extractor(target)
        return nn.functional.mse_loss(input_features, target_features)

def preprocess_image(img_array):
    # Convert to float and scale to [0, 1]
    img_tensor = torch.from_numpy(img_array).float() / 255.0
    
    # Permute dimensions from (H, W, C) to (C, H, W)
    img_tensor = img_tensor.permute(2, 0, 1)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor.to(device)
    
criterionPerceptual = PerceptualLoss().to(device)

for test_dir in test_dirs:
    # Extract sr value from directory name
    # sr = test_dir.split('sr')[-1]
    # print(f"\nProcessing degradation {blur}, sr {sr}")
    
    ssim_values = []
    psnr_values = []
    perceptual_values = []

    # Get original images
    original_files = [f for f in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, f)) and f.endswith(".jpg")]
    original_files.sort()

    print(f"{len(original_files)} images found in {original_dir}")

    # Create output text file
    output_path = os.path.join(test_dir, f'metrics_percep.txt')
    with open(output_path, 'w') as f:
        # f.write(f"Results for degradation {blur}, sr {sr}\n")
        f.write(f"Original directory: {original_dir}\n")
        f.write(f"Test directory: {test_dir}\n\n")

        # Process each image
        for file in original_files:
            original_path = os.path.join(original_dir, file)
            contrast_path = os.path.join(test_dir, file)
            
            if not os.path.exists(contrast_path):
                f.write(f"Contrast image not found for {file}\n")
                continue

            # Read images
            original = Image.open(original_path)
            contrast = Image.open(contrast_path)

            if original.size != contrast.size:
                contrast = contrast.resize(original.size)

            original = np.array(original)
            contrast = np.array(contrast)

            f.write(f"Processing: {file}\n")

            # Calculate SSIM
            ssim_value = compare_ssim(original, contrast, multichannel=True, win_size=3)
            ssim_values.append(ssim_value)
            f.write(f"SSIM: {ssim_value:.3f}\n")

            # Calculate PSNR
            psnr_value = compare_psnr(original, contrast)
            psnr_values.append(psnr_value)
            f.write(f"PSNR: {psnr_value:.3f} dB\n")

            # Preprocess images for perceptual loss
            original_tensor = preprocess_image(original)
            contrast_tensor = preprocess_image(contrast)

            # Calculate perceptual loss
            with torch.no_grad():
                loss_perceptual = criterionPerceptual(contrast_tensor, original_tensor)
            
            perceptual_values.append(loss_perceptual.item())
            f.write(f"Perceptual Loss: {loss_perceptual.item():.3f}\n\n")

        # Calculate and write mean values
        mean_ssim = round(np.mean(ssim_values), 3)
        mean_psnr = round(np.mean(psnr_values), 3)
        mean_perceptual = round(np.mean(perceptual_values), 3)

        f.write("=" * 50 + "\n")
        f.write(f"SUMMARY:\n")
        f.write(f"Mean SSIM: {mean_ssim}\n")
        f.write(f"Mean PSNR: {mean_psnr} dB\n")
        f.write(f"Mean Perceptual Loss: {mean_perceptual}\n")

    print(f"Results saved to {output_path}")
    print(f"Mean SSIM: {mean_ssim}")
    print(f"Mean PSNR: {mean_psnr} dB")
    print(f"Mean Perceptual Loss: {mean_perceptual}")

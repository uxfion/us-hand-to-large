#!/usr/bin/env python3
"""
Enhanced inference script for CycleGAN models.
Supports flexible image size settings, different model types, and various image formats.

Usage examples:
    python infer_new.py --name vq_dual_xijing_5090d --netG vq_dual --input_dir ./test_images --output_dir ./results
    python infer_new.py --name model_name --netG resnet_9blocks --input_path single_image.jpg --max_size 512
    python infer_new.py --name model_name --netG unet_256 --input_dir ./imgs --output_dir ./out --resize_base 32
"""

import torch
import torchvision.transforms as transforms
from models import create_model
from options.test_options import TestOptions
from data.base_dataset import get_transform
import numpy as np
from PIL import Image
import os
import argparse
from tqdm import tqdm
import sys
from pathlib import Path

class CycleGANInference:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load CycleGAN model with specified parameters"""
        # Build command line arguments for model loading
        model_args = [
            '--dataroot', self.args.dataroot,
            '--name', self.args.name,
            '--model', self.args.model,
            '--netG', self.args.netG,
            '--no_dropout',
            '--eval'
        ]
        
        # Set preprocessing based on mode
        if self.args.preprocess_mode == 'standard':
            model_args.extend(['--preprocess', 'resize_and_crop'])
            model_args.extend(['--load_size', str(self.args.load_size)])
            model_args.extend(['--crop_size', str(self.args.crop_size)])
        elif self.args.preprocess_mode == 'fixed_size':
            model_args.extend(['--preprocess', 'none'])  # We'll handle resize manually
        else:
            model_args.extend(['--preprocess', 'none'])
        
        # Add GPU settings
        if not self.args.cpu and torch.cuda.is_available():
            model_args.extend(['--gpu_ids', str(self.args.gpu_id)])
        else:
            model_args.extend(['--gpu_ids', '-1'])
            
        # Add channel settings
        model_args.extend(['--input_nc', str(self.args.input_nc)])
        model_args.extend(['--output_nc', str(self.args.output_nc)])
        
        # Add VQ-specific parameters if using VQ models
        if 'vq_dual' in self.args.netG.lower():
            model_args.extend(['--embed_dim', str(self.args.embed_dim)])
            model_args.extend(['--n_embed', str(self.args.n_embed)])
        elif 'contmix' in self.args.netG.lower():
            model_args.extend(['--embed_dim', str(self.args.embed_dim)])
            model_args.extend(['--n_embed', str(self.args.n_embed)])
            
        # Add model suffix if specified
        if self.args.model_suffix:
            model_args.extend(['--model_suffix', self.args.model_suffix])
            
        # Inject arguments into sys.argv
        original_argv = sys.argv.copy()
        sys.argv = ['infer_new.py'] + model_args
        
        try:
            # Parse options and create model
            opt = TestOptions().parse()
            opt.num_threads = 0
            opt.batch_size = 1
            opt.serial_batches = True
            opt.no_flip = True
            opt.display_id = -1
            
            # Create and setup model
            model = create_model(opt)
            model.setup(opt)
            model.eval()
            
            print(f"Model loaded successfully: {self.args.name}")
            print(f"Network: {self.args.netG}")
            print(f"Input channels: {self.args.input_nc}, Output channels: {self.args.output_nc}")
            print(f"Preprocessing mode: {self.args.preprocess_mode}")
            
            # Store the parsed options for use in preprocessing
            self.opt = opt
            
            return model
            
        finally:
            # Restore original sys.argv
            sys.argv = original_argv
            
    def _preprocess_image(self, image):
        """Preprocess image for model input using standard CycleGAN transforms"""
        original_size = image.size
        
        if self.args.preprocess_mode == 'fixed_size':
            # Custom fixed size preprocessing
            return self._fixed_size_preprocess(image, original_size)
        else:
            # Use standard framework preprocessing
            # Use the same transform that would be used in the dataset
            grayscale = (self.args.input_nc == 1)
            
            # For inference, we don't want random operations
            params = {'crop_pos': (0, 0), 'flip': False} if hasattr(self, 'opt') and 'crop' in self.opt.preprocess else None
            
            # Use standard transform from base_dataset
            transform = get_transform(self.opt, params=params, grayscale=grayscale, convert=True)
            
            # Apply transform
            tensor = transform(image).unsqueeze(0).to(self.device)
            
            if self.args.verbose:
                print(f"Original size: {original_size}, Processed tensor shape: {tensor.shape}")
                
            return tensor, original_size
            
    def _fixed_size_preprocess(self, image, original_size):
        """Fixed size preprocessing with aspect ratio preservation"""
        # Convert to appropriate mode first
        if self.args.input_nc == 1 and image.mode != 'L':
            image = image.convert('L')
        elif self.args.input_nc == 3 and image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize to fixed size while maintaining aspect ratio
        fixed_size = self.args.fixed_size
        w, h = image.size
        
        # Calculate scale to fit within fixed_size x fixed_size
        scale = min(fixed_size / w, fixed_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized_image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        
        # Create fixed size canvas and center the image
        if self.args.input_nc == 1:
            canvas = Image.new('L', (fixed_size, fixed_size), color=0)
        else:
            canvas = Image.new('RGB', (fixed_size, fixed_size), color=(0, 0, 0))
            
        # Center paste
        offset_x = (fixed_size - new_w) // 2
        offset_y = (fixed_size - new_h) // 2
        canvas.paste(resized_image, (offset_x, offset_y))
        
        # Apply transforms
        if self.args.input_nc == 1:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            
        tensor = transform(canvas).unsqueeze(0).to(self.device)
        
        if self.args.verbose:
            print(f"Fixed size preprocessing: {original_size} -> {canvas.size} -> tensor {tensor.shape}")
            print(f"Scale: {scale:.3f}, Offset: ({offset_x}, {offset_y})")
            
        # Store preprocessing info for later restoration
        preprocess_info = {
            'scale': scale,
            'offset': (offset_x, offset_y),
            'resized_size': (new_w, new_h)
        }
        
        return tensor, (original_size, preprocess_info)
        
    def _postprocess_output(self, output_tensor, original_info):
        """Postprocess model output to PIL image"""
        # Move to CPU and remove batch dimension
        output = output_tensor.cpu().squeeze(0)
        
        # Denormalize
        output = (output + 1) / 2.0
        output = torch.clamp(output, 0, 1)
        
        # Convert to PIL
        output_image = transforms.ToPILImage()(output)
        
        # Handle different preprocessing modes
        if self.args.preprocess_mode == 'fixed_size':
            # For fixed_size mode, restore to original size
            original_size, preprocess_info = original_info
            output_image = self._restore_from_fixed_size(output_image, original_size, preprocess_info)
        elif self.args.preprocess_mode == 'flexible':
            # For flexible mode with 'none' preprocessing, crop to original size if needed
            original_size = original_info
            current_size = output_image.size
            if current_size != original_size:
                output_image = output_image.crop((0, 0, min(original_size[0], current_size[0]), 
                                                min(original_size[1], current_size[1])))
                # If we need to resize back to exact original size
                if output_image.size != original_size:
                    output_image = output_image.resize(original_size, Image.Resampling.BICUBIC)
        else:
            # For standard mode, the image should already be the right size
            # But we may need to resize back to original size
            original_size = original_info
            if output_image.size != original_size:
                output_image = output_image.resize(original_size, Image.Resampling.BICUBIC)
        
        return output_image
        
    def _restore_from_fixed_size(self, output_image, original_size, preprocess_info):
        """Restore image from fixed size canvas to original size"""
        scale = preprocess_info['scale']
        offset_x, offset_y = preprocess_info['offset']
        resized_w, resized_h = preprocess_info['resized_size']
        
        # Extract the valid region from the fixed size canvas
        cropped = output_image.crop((offset_x, offset_y, offset_x + resized_w, offset_y + resized_h))
        
        # Resize back to original size
        restored = cropped.resize(original_size, Image.Resampling.BICUBIC)
        
        if self.args.verbose:
            print(f"Restored: {output_image.size} -> crop({offset_x},{offset_y},{offset_x + resized_w},{offset_y + resized_h}) -> resize to {original_size}")
        
        return restored
        
    def infer_single(self, image_path):
        """
        Perform inference on a single image
        
        Args:
            image_path: Path to input image
            
        Returns:
            PIL Image: Generated image
        """
        # Load image
        image = Image.open(image_path)
        
        # Preprocess
        input_tensor, original_info = self._preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            if hasattr(self.model, 'netG'):
                # For test model or similar
                if 'vq_dual' in self.args.netG and hasattr(self.model.netG, 'forward'):
                    output = self.model.netG(input_tensor, direction='AtoB')
                elif 'vq_resnet' in self.args.netG and hasattr(self.model.netG, 'forward'):
                    output, _ = self.model.netG(input_tensor)
                else:
                    output = self.model.netG(input_tensor)
            else:
                # For other model types
                self.model.set_input({'A': input_tensor, 'A_paths': [image_path]})
                self.model.test()
                visuals = self.model.get_current_visuals()
                output = visuals['fake']
                
        # Postprocess
        result_image = self._postprocess_output(output, original_info)
        
        return result_image
        
    def infer_batch(self, input_dir, output_dir):
        """
        Perform batch inference on a directory of images
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output images
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f'*{ext}'))
            image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
            
        if not image_files:
            print(f"No image files found in {input_dir}")
            return
            
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                # Generate output
                result_image = self.infer_single(image_path)
                
                # Save result
                output_path = Path(output_dir) / image_path.name
                result_image.save(output_path)
                
                if self.args.verbose:
                    print(f"Processed: {image_path.name}")
                    
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
                continue
                
        print(f"Batch processing completed. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced CycleGAN Inference")
    
    # Model settings
    parser.add_argument('--name', type=str, required=True, help='Model name (checkpoint folder)')
    parser.add_argument('--netG', type=str, default='resnet_9blocks', 
                       help='Generator architecture (resnet_9blocks, resnet_6blocks, unet_256, unet_128, vq_dual, etc.)')
    parser.add_argument('--model', type=str, default='test', 
                       help='Model type (test, cycle_gan, vq_test, etc.)')
    parser.add_argument('--model_suffix', type=str, default='', 
                       help='Model suffix for loading specific checkpoint')
    parser.add_argument('--dataroot', type=str, default='./datasets/placeholder', 
                       help='Placeholder dataroot (required by framework)')
    
    # Input/Output settings  
    parser.add_argument('--input_path', type=str, help='Single input image path')
    parser.add_argument('--input_dir', type=str, help='Input directory for batch processing')
    parser.add_argument('--output_dir', type=str, default='./inference_results', 
                       help='Output directory')
    parser.add_argument('--output_name', type=str, help='Output filename for single image')
    
    # Image processing settings
    parser.add_argument('--input_nc', type=int, default=1, help='Input image channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--output_nc', type=int, default=1, help='Output image channels (1 for grayscale, 3 for RGB)')
    
    # Preprocessing mode
    parser.add_argument('--preprocess_mode', type=str, default='flexible', 
                       choices=['flexible', 'standard', 'fixed_size'], 
                       help='flexible: power-of-2 padding; standard: resize+crop; fixed_size: resize to fixed size then restore')
    
    # Standard CycleGAN size options (for standard mode)
    parser.add_argument('--load_size', type=int, default=286, help='Scale images to this size (used in standard mode)')
    parser.add_argument('--crop_size', type=int, default=256, help='Crop to this size (used in standard mode)')
    
    # Fixed size mode options (for fixed_size mode)
    parser.add_argument('--fixed_size', type=int, default=512, help='Fixed size for inference (used in fixed_size mode)')
    
    # VQ model specific settings
    parser.add_argument('--embed_dim', type=int, default=3, help='VQ embedding dimension')
    parser.add_argument('--n_embed', type=int, default=1024, help='VQ number of embeddings')
    
    # Hardware settings
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    
    # Other settings
    parser.add_argument('--verbose', action='store_true', help='Print verbose information')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input_path and not args.input_dir:
        parser.error("Either --input_path or --input_dir must be specified")
        
    if args.input_path and args.input_dir:
        parser.error("Cannot specify both --input_path and --input_dir")
        
    if args.verbose:
        if args.preprocess_mode == 'standard':
            print(f"Using standard preprocessing: resize to {args.load_size}, crop to {args.crop_size}")
        elif args.preprocess_mode == 'fixed_size':
            print(f"Using fixed size preprocessing: resize to {args.fixed_size}x{args.fixed_size}, restore to original")
        else:
            print(f"Using flexible preprocessing: power-of-2 padding")
        
    # Create inference engine
    inference = CycleGANInference(args)
    
    # Run inference
    if args.input_path:
        # Single image inference
        if not os.path.exists(args.input_path):
            print(f"Input file not found: {args.input_path}")
            return
            
        result = inference.infer_single(args.input_path)
        
        # Determine output path
        if args.output_name:
            output_path = args.output_name
        else:
            input_name = Path(args.input_path).stem
            output_path = f"{args.output_dir}/{input_name}_generated.png"
            
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save result
        result.save(output_path)
        print(f"Result saved to: {output_path}")
        
    else:
        # Batch inference
        if not os.path.exists(args.input_dir):
            print(f"Input directory not found: {args.input_dir}")
            return
            
        inference.infer_batch(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()

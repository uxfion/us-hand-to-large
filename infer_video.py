#!/usr/bin/env python3
"""
Video processor with PIL-based image processing pipeline.
Uses ffmpeg for fast video I/O with Pillow for image processing.
Supports CycleGAN model integration for advanced image processing.
"""

import subprocess
import json
from PIL import Image, ImageFilter
from pathlib import Path
from typing import Callable, List, Optional
import time
import sys
import argparse

# Optional imports for CycleGAN support
try:
    import torch
    from infer_new import CycleGANInference
    CYCLEGAN_AVAILABLE = True
    print("CycleGAN dependencies available.")
except ImportError:
    CYCLEGAN_AVAILABLE = False
    torch = None
    CycleGANInference = None
    print("CycleGAN dependencies not available.")


class VideoProcessor:
    """Process videos frame-by-frame with PIL image processing pipeline."""

    def __init__(self, input_path: str, output_path: str = None):
        self.input_path = input_path
        self.output_path = output_path or self._generate_output_path()

        # Get video properties using ffprobe
        self.fps, self.width, self.height, self.total_frames = self._get_video_info()

    def _generate_output_path(self) -> str:
        """Generate output path based on input filename."""
        p = Path(self.input_path)
        return str(p.parent / f"{p.stem}_processed{p.suffix}")

    def _get_video_info(self):
        """Get video properties using ffprobe."""
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-count_frames',
            '-show_entries', 'stream=r_frame_rate,width,height,nb_read_frames',
            '-of', 'json',
            self.input_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            stream = info['streams'][0]

            # Parse frame rate
            fps_str = stream['r_frame_rate']
            num, den = map(int, fps_str.split('/'))
            fps = num / den

            width = stream['width']
            height = stream['height']
            total_frames = int(stream.get('nb_read_frames', 0))

            return fps, width, height, total_frames
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error: ffprobe failed. Please install ffmpeg.")
            print(f"On Ubuntu/Debian: sudo apt-get install ffmpeg")
            print(f"On macOS: brew install ffmpeg")
            raise

    def gaussian_blur(self, pil_image: Image.Image, radius: float = 2.0) -> Image.Image:
        """Apply Gaussian blur to PIL image."""
        return pil_image.filter(ImageFilter.GaussianBlur(radius=radius))

    def process_video(
        self,
        processing_functions: Optional[List[Callable[[Image.Image], Image.Image]]] = None,
        show_progress: bool = True
    ):
        """
        Process video with a list of PIL image processing functions.

        Args:
            processing_functions: List of functions that take and return PIL Images.
                                 If None, applies Gaussian blur by default.
            show_progress: Show processing progress.
        """
        if processing_functions is None:
            processing_functions = [self.gaussian_blur]

        # Start ffmpeg process to read frames as raw RGB
        input_cmd = [
            'ffmpeg', '-i', self.input_path,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo', '-'
        ]

        # Start ffmpeg process to write frames
        output_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'rgb24',
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            self.output_path
        ]

        frame_count = 0
        start_time = time.time()

        try:
            # Start both processes
            input_proc = subprocess.Popen(input_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            output_proc = subprocess.Popen(output_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

            frame_size = self.width * self.height * 3  # RGB24

            while True:
                # Read raw frame data
                raw_frame = input_proc.stdout.read(frame_size)

                if len(raw_frame) != frame_size:
                    break  # End of video

                # Convert raw bytes to PIL Image
                pil_frame = Image.frombytes('RGB', (self.width, self.height), raw_frame)

                # Apply all processing functions in sequence
                for func in processing_functions:
                    pil_frame = func(pil_frame)

                # Convert back to raw bytes and write
                output_proc.stdin.write(pil_frame.tobytes())

                frame_count += 1

                if show_progress and frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    if self.total_frames > 0:
                        progress = (frame_count / self.total_frames) * 100
                        print(f"Progress: {progress:.1f}% ({frame_count}/{self.total_frames}) - {fps:.1f} fps")
                    else:
                        print(f"Processed: {frame_count} frames - {fps:.1f} fps")

        finally:
            # Clean up
            if 'input_proc' in locals():
                input_proc.stdout.close()
                input_proc.wait()
            if 'output_proc' in locals():
                output_proc.stdin.close()
                output_proc.wait()

            if show_progress:
                elapsed = time.time() - start_time
                print(f"\nProcessing complete!")
                print(f"Total frames: {frame_count}")
                print(f"Total time: {elapsed:.2f}s")
                print(f"Average FPS: {frame_count/elapsed:.2f}")
                print(f"Output saved to: {self.output_path}")


# Example custom processing functions
def sharpen(pil_image: Image.Image) -> Image.Image:
    """Apply sharpening filter."""
    return pil_image.filter(ImageFilter.SHARPEN)


def edge_enhance(pil_image: Image.Image) -> Image.Image:
    """Apply edge enhancement filter."""
    return pil_image.filter(ImageFilter.EDGE_ENHANCE)


def custom_blur(radius: float = 5.0) -> Callable[[Image.Image], Image.Image]:
    """Create a custom blur function with specified radius."""
    def blur_func(pil_image: Image.Image) -> Image.Image:
        return pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
    return blur_func


def create_cyclegan_processor(args) -> Callable[[Image.Image], Image.Image]:
    """
    Create a CycleGAN processing function for video frames.

    Args:
        args: Namespace object with CycleGAN configuration (from argparse)

    Returns:
        Function that processes PIL Images using CycleGAN model
    """
    if not CYCLEGAN_AVAILABLE:
        raise ImportError("CycleGAN dependencies not available. Install: pip install torch torchvision tqdm")

    # Create CycleGAN inference instance
    cyclegan = CycleGANInference(args)

    # Create a processing function that uses the inference engine
    def process_frame(pil_image: Image.Image) -> Image.Image:
        """Process a single frame using CycleGAN model."""
        # Preprocess
        input_tensor, original_info = cyclegan._preprocess_image(pil_image)

        # Inference
        with torch.no_grad():
            if hasattr(cyclegan.model, 'netG'):
                if 'vq_dual' in args.netG and hasattr(cyclegan.model.netG, 'forward'):
                    output = cyclegan.model.netG(input_tensor, direction='AtoB')
                elif 'vq_resnet' in args.netG and hasattr(cyclegan.model.netG, 'forward'):
                    output, _ = cyclegan.model.netG(input_tensor)
                else:
                    output = cyclegan.model.netG(input_tensor)
            else:
                cyclegan.model.set_input({'A': input_tensor, 'A_paths': ['<video_frame>']})
                cyclegan.model.test()
                visuals = cyclegan.model.get_current_visuals()
                output = visuals['fake']

        # Postprocess
        result_image = cyclegan._postprocess_output(output, original_info)

        return result_image

    return process_frame


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Video processor with PIL and CycleGAN support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic Gaussian blur
  python video_processor.py input.mp4

  # CycleGAN processing
  python video_processor.py input.mp4 output.mp4 --use_cyclegan --name vq_dual_xijing_5090d --netG vq_dual

  # CycleGAN with fixed size preprocessing
  python video_processor.py input.mp4 output.mp4 --use_cyclegan --name my_model --netG resnet_9blocks --preprocess_mode fixed_size --fixed_size 512
        """
    )

    # Basic video processing arguments
    parser.add_argument('input_video', type=str, help='Input video file path')
    parser.add_argument('output_video', type=str, nargs='?', help='Output video file path (optional)')

    # CycleGAN mode flag
    parser.add_argument('--use_cyclegan', action='store_true', help='Use CycleGAN model for processing')

    # CycleGAN model settings
    parser.add_argument('--name', type=str, help='CycleGAN model name (checkpoint folder)')
    parser.add_argument('--netG', type=str, default='resnet_9blocks',
                       help='Generator architecture (resnet_9blocks, resnet_6blocks, unet_256, unet_128, vq_dual)')
    parser.add_argument('--model', type=str, default='test',
                       help='Model type (test, cycle_gan, vq_test)')
    parser.add_argument('--model_suffix', type=str, default='',
                       help='Model suffix for loading specific checkpoint')
    parser.add_argument('--dataroot', type=str, default='./datasets/placeholder',
                       help='Placeholder dataroot (required by framework)')

    # Image processing settings
    parser.add_argument('--input_nc', type=int, default=1,
                       help='Input image channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--output_nc', type=int, default=1,
                       help='Output image channels (1 for grayscale, 3 for RGB)')

    # Preprocessing mode
    parser.add_argument('--preprocess_mode', type=str, default='flexible',
                       choices=['flexible', 'standard', 'fixed_size'],
                       help='Preprocessing mode: flexible (default), standard (resize+crop), fixed_size (resize to fixed then restore)')

    # Standard CycleGAN size options
    parser.add_argument('--load_size', type=int, default=286,
                       help='Scale images to this size (used in standard mode)')
    parser.add_argument('--crop_size', type=int, default=256,
                       help='Crop to this size (used in standard mode)')

    # Fixed size mode options
    parser.add_argument('--fixed_size', type=int, default=512,
                       help='Fixed size for inference (used in fixed_size mode)')

    # VQ model specific settings
    parser.add_argument('--embed_dim', type=int, default=3,
                       help='VQ embedding dimension')
    parser.add_argument('--n_embed', type=int, default=1024,
                       help='VQ number of embeddings')

    # Hardware settings
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU instead of GPU')

    # Other settings
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose information')

    args = parser.parse_args()

    # Validate CycleGAN arguments
    if args.use_cyclegan:
        if not args.name:
            parser.error("--name is required when using --use_cyclegan")
        if not CYCLEGAN_AVAILABLE:
            print("Error: CycleGAN dependencies not available.")
            print("Install with: pip install torch torchvision tqdm")
            sys.exit(1)

    # Create processor
    processor = VideoProcessor(args.input_video, args.output_video)

    # Determine processing function
    if args.use_cyclegan:
        print(f"Using CycleGAN model: {args.name}")
        print(f"Generator: {args.netG}")
        print(f"Preprocessing: {args.preprocess_mode}")
        processing_func = create_cyclegan_processor(args)
        processor.process_video([processing_func])
    else:
        # Default: Simple Gaussian blur
        print("Applying Gaussian blur...")
        processor.process_video()

    # Example: Chain multiple processing functions
    # Uncomment to use:
    # print("Applying custom processing pipeline...")
    # processor.process_video([
    #     custom_blur(radius=3.0),
    #     edge_enhance,
    #     sharpen
    # ])
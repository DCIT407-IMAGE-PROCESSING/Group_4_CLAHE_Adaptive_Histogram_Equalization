"""
Contrast Limited Adaptive Histogram Equalization (CLAHE) Implementation
Author: GROUP 4

This module implements CLAHE for image enhancement.
CLAHE applies histogram equalization to local regions (tiles) and limits
contrast amplification to prevent noise over-enhancement.
"""

import cv2
import numpy as np
import json
import os
import glob
import argparse
from typing import Tuple, Dict, List
from pathlib import Path
import time


def apply_clahe_grayscale(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE to a grayscale image.

    Args:
        image: Input grayscale image (H x W)
        clip_limit: Threshold for contrast limiting (default: 2.0)
        tile_grid_size: Size of grid for histogram equalization (default: 8x8)

    Returns:
        Enhanced grayscale image
    """
    if len(image.shape) != 2:
        raise ValueError("Input must be a grayscale image")

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE
    enhanced = clahe.apply(image)

    return enhanced


def apply_clahe_color(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    color_space: str = 'LAB'
) -> np.ndarray:
    """
    Apply CLAHE to a color image.

    Args:
        image: Input BGR color image (H x W x 3)
        clip_limit: Threshold for contrast limiting (default: 2.0)
        tile_grid_size: Size of grid for histogram equalization (default: 8x8)
        color_space: Color space for equalization ('LAB', 'HSV', or 'YCrCb')

    Returns:
        Enhanced BGR color image
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a BGR color image")

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Convert to appropriate color space
    if color_space == 'LAB':
        img_converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # Apply CLAHE only to the L channel (lightness)
        img_converted[:, :, 0] = clahe.apply(img_converted[:, :, 0])
        # Convert back to BGR
        enhanced = cv2.cvtColor(img_converted, cv2.COLOR_LAB2BGR)

    elif color_space == 'HSV':
        img_converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Apply CLAHE only to the V channel (value/brightness)
        img_converted[:, :, 2] = clahe.apply(img_converted[:, :, 2])
        # Convert back to BGR
        enhanced = cv2.cvtColor(img_converted, cv2.COLOR_HSV2BGR)

    elif color_space == 'YCrCb':
        img_converted = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # Apply CLAHE only to the Y channel (luminance)
        img_converted[:, :, 0] = clahe.apply(img_converted[:, :, 0])
        # Convert back to BGR
        enhanced = cv2.cvtColor(img_converted, cv2.COLOR_YCrCb2BGR)

    else:
        raise ValueError(f"Unsupported color space: {color_space}")

    return enhanced


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    color_space: str = 'LAB',
    grayscale: bool = False
) -> np.ndarray:
    """
    Apply CLAHE to an image.
    Automatically detects if image is grayscale or color.

    Args:
        image: Input image (grayscale or BGR color)
        clip_limit: Threshold for contrast limiting (default: 2.0)
        tile_grid_size: Size of grid for histogram equalization (default: 8x8)
        color_space: Color space for color images ('LAB', 'HSV', or 'YCrCb')
        grayscale: Force grayscale processing

    Returns:
        Enhanced image
    """
    if grayscale and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if len(image.shape) == 2:
        return apply_clahe_grayscale(image, clip_limit, tile_grid_size)
    elif len(image.shape) == 3:
        return apply_clahe_color(image, clip_limit, tile_grid_size, color_space)
    else:
        raise ValueError("Invalid image shape")


def calculate_metrics(original: np.ndarray, enhanced: np.ndarray) -> Dict:
    """
    Calculate quality metrics for enhanced image.

    Args:
        original: Original input image
        enhanced: Enhanced output image

    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Check if images are grayscale or color
    is_original_gray = len(original.shape) == 2
    is_enhanced_gray = len(enhanced.shape) == 2
    
    # Convert to grayscale for metrics if needed
    if is_original_gray:
        orig_gray = original
    else:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    if is_enhanced_gray:
        enh_gray = enhanced
    else:
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    
    # Entropy
    hist = cv2.calcHist([enh_gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    metrics['entropy'] = float(-np.sum(hist * np.log2(hist)))
    
    # Contrast (RMS)
    metrics['contrast'] = float(enh_gray.std())
    
    # Brightness
    metrics['brightness'] = float(enh_gray.mean())
    
    # Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(enh_gray, cv2.CV_64F)
    metrics['sharpness'] = float(laplacian.var())
    
    # PSNR - need to ensure both images have same shape
    if original.shape != enhanced.shape:
        # If grayscale processing was applied, compare gray versions
        if is_enhanced_gray and not is_original_gray:
            mse = np.mean((orig_gray.astype(float) - enhanced.astype(float)) ** 2)
        else:
            # Resize if necessary
            enhanced_resized = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
            mse = np.mean((original.astype(float) - enhanced_resized.astype(float)) ** 2)
    else:
        mse = np.mean((original.astype(float) - enhanced.astype(float)) ** 2)
    
    if mse == 0:
        metrics['psnr'] = float('inf')
    else:
        max_pixel = 255.0
        metrics['psnr'] = float(20 * np.log10(max_pixel / np.sqrt(mse)))
    
    # Image dimensions
    metrics['width'] = int(enhanced.shape[1])
    metrics['height'] = int(enhanced.shape[0])
    metrics['channels'] = int(enhanced.shape[2]) if len(enhanced.shape) == 3 else 1
    
    return metrics


def tune_clahe_parameters(
    image: np.ndarray,
    clip_limits: List[float] = [1.0, 2.0, 3.0, 4.0],
    tile_sizes: List[Tuple[int, int]] = [(4, 4), (8, 8), (16, 16)],
    color_spaces: List[str] = ['LAB', 'HSV', 'YCrCb']
) -> Dict:
    """
    Experiment with different CLAHE parameters.

    Args:
        image: Input image
        clip_limits: List of clip limit values to try
        tile_sizes: List of tile grid sizes to try
        color_spaces: List of color spaces to try

    Returns:
        Dictionary of results with different parameter combinations
    """
    results = {}
    is_grayscale = len(image.shape) == 2

    for clip_limit in clip_limits:
        for tile_size in tile_sizes:
            if is_grayscale:
                key = f"clip_{clip_limit}_tile_{tile_size[0]}x{tile_size[1]}"
                enhanced = apply_clahe(image, clip_limit, tile_size)
                results[key] = {
                    'image': enhanced,
                    'clip_limit': clip_limit,
                    'tile_size': tile_size,
                    'color_space': 'grayscale'
                }
            else:
                for color_space in color_spaces:
                    key = f"clip_{clip_limit}_tile_{tile_size[0]}x{tile_size[1]}_{color_space}"
                    enhanced = apply_clahe(image, clip_limit, tile_size, color_space)
                    results[key] = {
                        'image': enhanced,
                        'clip_limit': clip_limit,
                        'tile_size': tile_size,
                        'color_space': color_space
                    }

    return results


def process_single_image(
    input_path: str,
    output_dir: str,
    clip_limit: float = 2.0,
    tile_size: int = 8,
    color_space: str = 'LAB',
    grayscale: bool = False,
    show: bool = False
) -> Dict:
    """
    Process a single image with CLAHE.

    Args:
        input_path: Path to input image
        output_dir: Directory to save output
        clip_limit: CLAHE clip limit
        tile_size: CLAHE tile grid size
        color_space: Color space for processing
        grayscale: Process as grayscale
        show: Display result

    Returns:
        Dictionary with processing results and metrics
    """
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not load image: {input_path}")
    
    # Store original for metrics
    original = image.copy()
    
    # Apply CLAHE
    start_time = time.time()
    enhanced = apply_clahe(
        image,
        clip_limit=clip_limit,
        tile_grid_size=(tile_size, tile_size),
        color_space=color_space,
        grayscale=grayscale
    )
    processing_time = time.time() - start_time
    
    # Generate output filename
    input_filename = Path(input_path).stem
    if grayscale:
        output_filename = f"{input_filename}_clahe_clip{clip_limit}_tile{tile_size}_gray.png"
    else:
        output_filename = f"{input_filename}_clahe_clip{clip_limit}_tile{tile_size}_{color_space}.png"
    
    output_path = os.path.join(output_dir, output_filename)
    
    # Save enhanced image
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, enhanced)
    
    # Calculate metrics
    metrics = calculate_metrics(original, enhanced)
    
    # Add processing info
    result = {
        'input_path': input_path,
        'output_path': output_path,
        'parameters': {
            'clip_limit': clip_limit,
            'tile_size': tile_size,
            'color_space': color_space if not grayscale else 'grayscale',
            'grayscale': grayscale
        },
        'metrics': metrics,
        'processing_time_seconds': processing_time
    }
    
    # Display if requested
    if show:
        # Prepare images for display
        if len(enhanced.shape) == 2:
            # Grayscale - convert to BGR for display
            enhanced_display = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            enhanced_display = enhanced
        
        # Resize for display
        display = np.hstack([
            cv2.resize(original, (400, 400)), 
            cv2.resize(enhanced_display, (400, 400))
        ])
        cv2.imshow(f'Original vs CLAHE - {input_filename}', display)
        print("\nPress any key to close the display window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result


def process_batch(
    input_dir: str = 'data/input',
    output_dir: str = 'data/output',
    test_parameters: bool = True
) -> List[Dict]:
    """
    Process all images in input directory with multiple CLAHE configurations.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save outputs
        test_parameters: If True, test multiple parameter combinations

    Returns:
        List of results for all processed images
    """
    all_results = []
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return all_results
    
    print(f"Found {len(image_files)} images to process")
    
    if test_parameters:
        # Test multiple parameter combinations
        configs = [
            {'clip_limit': 2.0, 'tile_size': 8, 'color_space': 'LAB', 'grayscale': False},
            {'clip_limit': 3.0, 'tile_size': 16, 'color_space': 'HSV', 'grayscale': False},
            {'clip_limit': 2.0, 'tile_size': 8, 'color_space': 'YCrCb', 'grayscale': False},
            {'clip_limit': 4.0, 'tile_size': 8, 'color_space': 'LAB', 'grayscale': False},
            {'clip_limit': 2.0, 'tile_size': 8, 'color_space': 'LAB', 'grayscale': True},
        ]
    else:
        # Default configuration only
        configs = [
            {'clip_limit': 2.0, 'tile_size': 8, 'color_space': 'LAB', 'grayscale': False}
        ]
    
    # Process each image with each configuration
    for img_path in image_files:
        print(f"\nProcessing: {os.path.basename(img_path)}")
        
        for config in configs:
            try:
                result = process_single_image(
                    input_path=img_path,
                    output_dir=output_dir,
                    **config,
                    show=False
                )
                all_results.append(result)
                print(f"  ✓ clip={config['clip_limit']}, tile={config['tile_size']}, "
                      f"space={config['color_space']}, gray={config['grayscale']}")
            except Exception as e:
                print(f"  ✗ Error with config {config}: {str(e)}")
    
    return all_results


def save_metrics_to_json(results: List[Dict], output_path: str = 'metrics.json'):
    """
    Save all results and metrics to a JSON file.

    Args:
        results: List of processing results
        output_path: Path to save JSON file
    """
    # Prepare summary
    summary = {
        'total_images_processed': len(results),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': results
    }
    
    # Calculate average metrics
    if results:
        avg_metrics = {}
        metric_keys = list(results[0]['metrics'].keys())
        for key in metric_keys:
            values = [r['metrics'][key] for r in results if key in r['metrics']]
            if values and not np.isinf(values[0]):
                avg_metrics[f'avg_{key}'] = float(np.mean(values))
        
        summary['average_metrics'] = avg_metrics
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CLAHE Image Enhancement - Standalone Script'
    )
    
    # Input/Output
    parser.add_argument('--input', type=str, help='Path to input image')
    parser.add_argument('--input-dir', type=str, default='data/input',
                       help='Input directory for batch processing')
    parser.add_argument('--output', type=str, default='data/output',
                       help='Output directory')
    
    # CLAHE Parameters
    parser.add_argument('--clip-limit', type=float, default=2.0,
                       help='CLAHE clip limit (default: 2.0)')
    parser.add_argument('--tile-size', type=int, default=8,
                       help='CLAHE tile grid size (default: 8)')
    parser.add_argument('--color-space', type=str, default='LAB',
                       choices=['LAB', 'HSV', 'YCrCb'],
                       help='Color space for processing')
    
    # Options
    parser.add_argument('--grayscale', action='store_true',
                       help='Process as grayscale')
    parser.add_argument('--show', action='store_true',
                       help='Display results')
    parser.add_argument('--test-params', action='store_true',
                       help='Test multiple parameter combinations')
    parser.add_argument('--metrics-output', type=str, default='metrics.json',
                       help='Path to save metrics JSON file')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CLAHE - Contrast Limited Adaptive Histogram Equalization")
    print("Author: GROUP 4")
    print("=" * 60)
    
    results = []
    
    # Single image processing
    if args.input:
        print(f"\n→ Processing single image: {args.input}")
        result = process_single_image(
            input_path=args.input,
            output_dir=args.output,
            clip_limit=args.clip_limit,
            tile_size=args.tile_size,
            color_space=args.color_space,
            grayscale=args.grayscale,
            show=args.show
        )
        results.append(result)
        
        print(f"\n✓ Output saved to: {result['output_path']}")
        print("\nMetrics:")
        for key, value in result['metrics'].items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Batch processing
    else:
        print(f"\n→ Batch processing from: {args.input_dir}")
        results = process_batch(
            input_dir=args.input_dir,
            output_dir=args.output,
            test_parameters=args.test_params
        )
    
    # Save metrics to JSON
    if results:
        save_metrics_to_json(results, args.metrics_output)
        print(f"\n✓ Total images processed: {len(results)}")
    else:
        print("\n✗ No images were processed")
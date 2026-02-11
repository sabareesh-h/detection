"""
Image Quality Validation Script
Validates captured images before adding to training dataset
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


class ImageQualityValidator:
    """Validates image quality for defect detection training"""
    
    def __init__(self, 
                 min_resolution: Tuple[int, int] = (640, 480),
                 brightness_range: Tuple[int, int] = (50, 205),
                 min_contrast: float = 20.0,
                 min_sharpness: float = 100.0):
        """
        Initialize validator with quality thresholds
        
        Args:
            min_resolution: Minimum (width, height)
            brightness_range: Acceptable (min, max) mean brightness
            min_contrast: Minimum standard deviation for contrast
            min_sharpness: Minimum Laplacian variance for sharpness
        """
        self.min_resolution = min_resolution
        self.brightness_range = brightness_range
        self.min_contrast = min_contrast
        self.min_sharpness = min_sharpness
    
    def validate(self, image_path: str) -> Dict:
        """
        Validate a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict with validation results and metrics
        """
        img = cv2.imread(image_path)
        
        if img is None:
            return {
                'path': image_path,
                'passed': False,
                'error': 'Failed to load image',
                'issues': ['Cannot read image file']
            }
        
        issues = []
        metrics = {}
        
        # Check 1: Resolution
        height, width = img.shape[:2]
        metrics['resolution'] = (width, height)
        if width < self.min_resolution[0] or height < self.min_resolution[1]:
            issues.append(f"Low resolution: {width}x{height} (min: {self.min_resolution[0]}x{self.min_resolution[1]})")
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Check 2: Brightness (mean pixel value)
        mean_brightness = np.mean(gray)
        metrics['brightness'] = round(mean_brightness, 2)
        if mean_brightness < self.brightness_range[0]:
            issues.append(f"Too dark: brightness={mean_brightness:.1f} (min: {self.brightness_range[0]})")
        elif mean_brightness > self.brightness_range[1]:
            issues.append(f"Too bright: brightness={mean_brightness:.1f} (max: {self.brightness_range[1]})")
        
        # Check 3: Contrast (standard deviation)
        std_contrast = np.std(gray)
        metrics['contrast'] = round(std_contrast, 2)
        if std_contrast < self.min_contrast:
            issues.append(f"Low contrast: std={std_contrast:.1f} (min: {self.min_contrast})")
        
        # Check 4: Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['sharpness'] = round(laplacian_var, 2)
        if laplacian_var < self.min_sharpness:
            issues.append(f"Blurry: sharpness={laplacian_var:.1f} (min: {self.min_sharpness})")
        
        # Check 5: Clipping (over/under exposure)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        pixel_count = gray.size
        underexposed = hist[0][0] / pixel_count
        overexposed = hist[255][0] / pixel_count
        metrics['underexposed_pct'] = round(underexposed * 100, 2)
        metrics['overexposed_pct'] = round(overexposed * 100, 2)
        
        if underexposed > 0.1:
            issues.append(f"Underexposed: {underexposed*100:.1f}% pixels clipped at black")
        if overexposed > 0.1:
            issues.append(f"Overexposed: {overexposed*100:.1f}% pixels clipped at white")
        
        return {
            'path': image_path,
            'passed': len(issues) == 0,
            'issues': issues,
            'metrics': metrics
        }
    
    def validate_directory(self, directory: str, 
                          extensions: List[str] = ['png', 'jpg', 'jpeg']) -> Dict:
        """
        Validate all images in a directory
        
        Args:
            directory: Path to directory with images
            extensions: List of file extensions to check
            
        Returns:
            Summary dict with pass/fail counts and problem images
        """
        results = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'problem_images': []
        }
        
        # Find all images
        patterns = [f"{directory}/**/*.{ext}" for ext in extensions]
        image_paths = []
        for pattern in patterns:
            image_paths.extend(glob.glob(pattern, recursive=True))
        
        print(f"Found {len(image_paths)} images to validate...")
        
        for img_path in image_paths:
            result = self.validate(img_path)
            results['total'] += 1
            
            if result['passed']:
                results['passed'] += 1
            else:
                results['failed'] += 1
                results['problem_images'].append(result)
        
        return results
    
    def print_report(self, directory: str):
        """Print a formatted validation report for a directory"""
        results = self.validate_directory(directory)
        
        print("\n" + "="*60)
        print("IMAGE QUALITY VALIDATION REPORT")
        print("="*60)
        print(f"Directory: {directory}")
        print(f"Total images: {results['total']}")
        print(f"Passed: {results['passed']} ({results['passed']/max(1,results['total'])*100:.1f}%)")
        print(f"Failed: {results['failed']} ({results['failed']/max(1,results['total'])*100:.1f}%)")
        
        if results['problem_images']:
            print("\n" + "-"*60)
            print("PROBLEM IMAGES:")
            print("-"*60)
            for prob in results['problem_images'][:10]:  # Show first 10
                print(f"\n  {Path(prob['path']).name}")
                for issue in prob['issues']:
                    print(f"    ⚠ {issue}")
            
            if len(results['problem_images']) > 10:
                print(f"\n  ... and {len(results['problem_images'])-10} more problem images")
        
        print("\n" + "="*60)
        return results


def main():
    """Run quality validation on dataset directory"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate image quality for training')
    parser.add_argument('directory', nargs='?', default='dataset/raw',
                       help='Directory containing images to validate')
    parser.add_argument('--min-width', type=int, default=640,
                       help='Minimum image width')
    parser.add_argument('--min-height', type=int, default=480,
                       help='Minimum image height')
    parser.add_argument('--min-sharpness', type=float, default=100.0,
                       help='Minimum sharpness threshold')
    
    args = parser.parse_args()
    
    validator = ImageQualityValidator(
        min_resolution=(args.min_width, args.min_height),
        min_sharpness=args.min_sharpness
    )
    
    validator.print_report(args.directory)


if __name__ == "__main__":
    main()

"""
=============================================================
  validate_images.py  --  Defect detection pipeline script
=============================================================
HOW TO USE
----------
python validate_images.py [-h] [--min-width MIN_WIDTH] [--check-integrity] [--data DATA_YAML]

FLAGS
-----
-h, --help            show this help message and exit
    --min-width MIN_WIDTH
    Minimum image width
    --min-height MIN_HEIGHT
    Minimum image height
    --min-sharpness MIN_SHARPNESS
    Minimum sharpness threshold
    --check-integrity
    Run strict YOLO dataset integrity checks (orphans, coordinates)
    --data DATA_YAML
    Path to config/dataset.yaml (used with --check-integrity)
=============================================================
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import hashlib

class ImageQualityValidator:
    """Validates image quality for defect detection training"""
    
    def __init__(self, 
                 min_resolution: Tuple[int, int] = (640, 480),
                 brightness_range: Tuple[int, int] = (50, 205),
                 min_contrast: float = 20.0,
                 min_sharpness: float = 100.0):
        self.min_resolution = min_resolution
        self.brightness_range = brightness_range
        self.min_contrast = min_contrast
        self.min_sharpness = min_sharpness
    
    def validate(self, image_path: str) -> Dict:
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
        
        # Check 2: Brightness
        mean_brightness = np.mean(gray)
        metrics['brightness'] = round(mean_brightness, 2)
        if mean_brightness < self.brightness_range[0]:
            issues.append(f"Too dark: brightness={mean_brightness:.1f} (min: {self.brightness_range[0]})")
        elif mean_brightness > self.brightness_range[1]:
            issues.append(f"Too bright: brightness={mean_brightness:.1f} (max: {self.brightness_range[1]})")
        
        # Check 3: Contrast
        std_contrast = np.std(gray)
        metrics['contrast'] = round(std_contrast, 2)
        if std_contrast < self.min_contrast:
            issues.append(f"Low contrast: std={std_contrast:.1f} (min: {self.min_contrast})")
        
        # Check 4: Sharpness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['sharpness'] = round(laplacian_var, 2)
        if laplacian_var < self.min_sharpness:
            issues.append(f"Blurry: sharpness={laplacian_var:.1f} (min: {self.min_sharpness})")
        
        # Check 5: Clipping
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
        results = {'total': 0, 'passed': 0, 'failed': 0, 'problem_images': []}
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
            for prob in results['problem_images'][:10]:
                print(f"\n  {Path(prob['path']).name}")
                for issue in prob['issues']:
                    print(f"    ⚠ {issue}")
            if len(results['problem_images']) > 10:
                print(f"\n  ... and {len(results['problem_images'])-10} more problem images")
        print("\n" + "="*60)


class DatasetIntegrityValidator:
    """Validates dataset integrity for YOLO training (orphans, coordinates, etc.)"""
    
    def __init__(self, dataset_yaml_path: str):
        self.dataset_yaml_path = Path(dataset_yaml_path)
        self.nc = 0
        self.class_names = {}
        self._load_yaml()
        
    def _load_yaml(self):
        if not self.dataset_yaml_path.exists():
            print(f"Warning: Dataset config {self.dataset_yaml_path} not found. Assuming nc=1.")
            self.nc = 1
            return
            
        with open(self.dataset_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            self.nc = data.get('nc', 1)
            self.class_names = data.get('names', {})
            if isinstance(self.class_names, list):
                self.class_names = {i: name for i, name in enumerate(self.class_names)}
            print(f"Loaded dataset config: nc={self.nc}")
            
    def validate_dataset(self, dataset_dir: str):
        print("\n" + "="*60)
        print("DATASET INTEGRITY VALIDATION REPORT")
        print("="*60)
        dataset_path = Path(dataset_dir)
        
        splits = ['train', 'val', 'test']
        all_issues = []
        image_hashes = {}
        class_counts = {i: 0 for i in range(self.nc)}
        total_labels = 0
        
        for split in splits:
            images_dir = dataset_path / "images" / split
            labels_dir = dataset_path / "labels" / split
            
            if not images_dir.exists():
                continue
                
            print(f"Scanning {split} split...")
            images = list(images_dir.glob("*.*"))
            labels = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
            
            image_stems = {img.stem: img for img in images}
            label_stems = {lbl.stem: lbl for lbl in labels}
            
            # 1. Orphan Checks & Image validation
            for stem, img_path in image_stems.items():
                if stem not in label_stems:
                    all_issues.append(f"Orphan Image: {split}/{img_path.name} has no corresponding .txt label")
                
                # Zero-byte check
                if img_path.stat().st_size == 0:
                    all_issues.append(f"Corrupt Image: {split}/{img_path.name} is 0 bytes")
                    
                # Hash check for duplicates
                try:
                    with open(img_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                        if file_hash in image_hashes:
                            all_issues.append(f"Duplicate Image: {split}/{img_path.name} is identical to {image_hashes[file_hash]}")
                        else:
                            image_hashes[file_hash] = f"{split}/{img_path.name}"
                except Exception as e:
                    pass

            # 2. Label validation
            for stem, lbl_path in label_stems.items():
                if stem not in image_stems:
                    all_issues.append(f"Orphan Label: {split}/{lbl_path.name} has no corresponding image")
                    
                if lbl_path.stat().st_size == 0:
                    continue # empty label means no objects, which is valid in YOLO for background images
                    
                # Parse label contents
                with open(lbl_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if not parts:
                            continue
                        
                        try:
                            class_id = int(parts[0])
                            total_labels += 1
                            
                            # Valid Class ID check
                            if class_id < 0 or class_id >= self.nc:
                                all_issues.append(f"Invalid Class ID: {split}/{lbl_path.name} line {line_num} has class {class_id} (max allowed: {self.nc-1})")
                            else:
                                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                                
                            # Coordinate bounds check
                            coords = [float(x) for x in parts[1:]]
                            for c in coords:
                                if c < 0.0 or c > 1.0:
                                    all_issues.append(f"Out-of-bounds Coordinate: {split}/{lbl_path.name} line {line_num} has coordinate {c} (must be 0.0 to 1.0)")
                                    break
                                    
                        except ValueError:
                            all_issues.append(f"Malformed Label: {split}/{lbl_path.name} line {line_num} is unreadable")

        # Report results
        print("\n")
        if not all_issues:
            print("[PASSED] Dataset Integrity: No errors found.")
        else:
            print(f"[FAILED] Dataset Integrity: {len(all_issues)} issues found:")
            for issue in all_issues[:20]:
                print(f"  - {issue}")
            if len(all_issues) > 20:
                print(f"  ... and {len(all_issues) - 20} more issues")
                
        # Class Distribution Report
        print("\n--- Class Distribution ---")
        if total_labels == 0:
            print("No labels found.")
        else:
            for cls_id in range(self.nc):
                count = class_counts.get(cls_id, 0)
                pct = (count / total_labels) * 100 if total_labels > 0 else 0
                name = self.class_names.get(cls_id, f"Class {cls_id}")
                warning = " [WARNING: UNDERREPRESENTED (<10%)]" if pct < 10.0 else ""
                print(f"  Class {cls_id} ({name}): {count} labels ({pct:.1f}%){warning}")
        
        print("="*60)
        return len(all_issues) == 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate image quality and dataset integrity for training')
    parser.add_argument('directory', nargs='?', default='dataset',
                       help='Directory containing dataset to validate')
    
    # Quality flags
    parser.add_argument('--min-width', type=int, default=640, help='Minimum image width')
    parser.add_argument('--min-height', type=int, default=480, help='Minimum image height')
    parser.add_argument('--min-sharpness', type=float, default=100.0, help='Minimum sharpness threshold')
    
    # Integrity flags
    parser.add_argument('--check-integrity', action='store_true', help='Run strict dataset integrity validation (orphans, coordinates, classes)')
    parser.add_argument('--data', type=str, default='../config/dataset.yaml', help='Path to dataset.yaml (for class config)')
    
    args = parser.parse_args()
    
    if args.check_integrity:
        yaml_path = args.data
        if not Path(yaml_path).exists():
            # Try finding it in config/
            alt_path = Path("config/dataset.yaml")
            if alt_path.exists():
                yaml_path = str(alt_path)
            else:
                alt_path2 = Path("../config/dataset.yaml")
                if alt_path2.exists():
                    yaml_path = str(alt_path2)
                    
        integrity_validator = DatasetIntegrityValidator(dataset_yaml_path=yaml_path)
        integrity_validator.validate_dataset(args.directory)
    else:
        # Run quality validator (default)
        # If running from scripts, default directory should be dataset/raw
        target_dir = args.directory
        if target_dir == 'dataset' and Path('dataset/raw').exists():
            target_dir = 'dataset/raw'
            
        validator = ImageQualityValidator(
            min_resolution=(args.min_width, args.min_height),
            min_sharpness=args.min_sharpness
        )
        validator.print_report(target_dir)

if __name__ == "__main__":
    main()

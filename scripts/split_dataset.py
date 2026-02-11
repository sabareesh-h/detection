"""
Dataset Splitting Utility
Splits raw dataset into train/val/test sets with stratified sampling
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, List
import random

try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    copy_files: bool = True
) -> dict:
    """
    Split dataset into train/val/test sets maintaining class distribution
    
    Args:
        source_dir: Directory with class subfolders (e.g., raw/good, raw/scratch)
        output_dir: Output directory for split dataset
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set  
        test_ratio: Fraction for test set
        random_seed: Random seed for reproducibility
        copy_files: If True, copy files; if False, move files
        
    Returns:
        dict with split statistics
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
        "Ratios must sum to 1.0"
    
    random.seed(random_seed)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total': 0,
        'train': 0,
        'val': 0,
        'test': 0,
        'classes': {}
    }
    
    # Process each class directory
    for class_dir in source_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        
        # Get all images in this class
        images = list(class_dir.glob('*.png')) + \
                 list(class_dir.glob('*.jpg')) + \
                 list(class_dir.glob('*.jpeg'))
        
        if len(images) == 0:
            print(f"Warning: No images found in {class_dir}")
            continue
        
        # Shuffle and split
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Copy/move files
        splits = [
            ('train', train_images),
            ('val', val_images),
            ('test', test_images)
        ]
        
        for split_name, split_images in splits:
            for img_path in split_images:
                # Destination for image
                dest_img = output_path / 'images' / split_name / img_path.name
                
                if copy_files:
                    shutil.copy2(img_path, dest_img)
                else:
                    shutil.move(img_path, dest_img)
                
                # Check for corresponding label file
                label_path = img_path.with_suffix('.txt')
                if label_path.exists():
                    dest_label = output_path / 'labels' / split_name / label_path.name
                    if copy_files:
                        shutil.copy2(label_path, dest_label)
                    else:
                        shutil.move(label_path, dest_label)
        
        # Update statistics
        stats['total'] += n_total
        stats['train'] += len(train_images)
        stats['val'] += len(val_images)
        stats['test'] += len(test_images)
        stats['classes'][class_name] = {
            'total': n_total,
            'train': len(train_images),
            'val': len(val_images),
            'test': len(test_images)
        }
        
        print(f"{class_name}: {n_total} images -> "
              f"train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")
    
    return stats


def print_split_summary(stats: dict):
    """Print a formatted summary of the dataset split"""
    print("\n" + "="*60)
    print("DATASET SPLIT SUMMARY")
    print("="*60)
    print(f"Total images: {stats['total']}")
    print(f"  Train: {stats['train']} ({stats['train']/max(1,stats['total'])*100:.1f}%)")
    print(f"  Val:   {stats['val']} ({stats['val']/max(1,stats['total'])*100:.1f}%)")
    print(f"  Test:  {stats['test']} ({stats['test']/max(1,stats['total'])*100:.1f}%)")
    
    print("\nPer-class breakdown:")
    print("-"*60)
    for class_name, class_stats in stats['classes'].items():
        print(f"  {class_name}:")
        print(f"    Total: {class_stats['total']}, "
              f"Train: {class_stats['train']}, "
              f"Val: {class_stats['val']}, "
              f"Test: {class_stats['test']}")
    print("="*60)


def main():
    """Run dataset splitting from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Split dataset for YOLO training')
    parser.add_argument('source', nargs='?', default='dataset/raw',
                       help='Source directory with class subfolders')
    parser.add_argument('--output', '-o', default='dataset',
                       help='Output directory for split dataset')
    parser.add_argument('--train', type=float, default=0.70,
                       help='Training set ratio (default: 0.70)')
    parser.add_argument('--val', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--move', action='store_true',
                       help='Move files instead of copying')
    
    args = parser.parse_args()
    
    print(f"Splitting dataset from: {args.source}")
    print(f"Output directory: {args.output}")
    print(f"Ratios: train={args.train}, val={args.val}, test={args.test}")
    
    stats = split_dataset(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        random_seed=args.seed,
        copy_files=not args.move
    )
    
    print_split_summary(stats)


if __name__ == "__main__":
    main()

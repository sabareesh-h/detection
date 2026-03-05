"""
YOLOv11m Training Script for Defect Detection
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import torch

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Error: ultralytics not installed. Run: pip install ultralytics")


def check_environment():
    """Check and print environment information"""
    print("="*60)
    print("ENVIRONMENT CHECK")
    print("="*60)
    
    # Python version
    print(f"Python: {sys.version}")
    
    # PyTorch and CUDA
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected! Training will be very slow on CPU.")
    
    # Ultralytics
    if ULTRALYTICS_AVAILABLE:
        from ultralytics import __version__ as ultra_version
        print(f"Ultralytics: {ultra_version}")
    
    print("="*60 + "\n")
    
    return torch.cuda.is_available()


def train_yolov11m(
    data_config: str = "config/dataset.yaml",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    patience: int = 20,
    device: int = 0,
    project: str = "models",
    name: str = None,
    resume: bool = False,
    weights: str = "yolo26m.pt"
):
    """
    Train YOLOv11m model on defect detection dataset
    
    Args:
        data_config: Path to dataset.yaml configuration
        epochs: Number of training epochs
        batch_size: Batch size (reduce if GPU memory issues)
        img_size: Input image size
        patience: Early stopping patience
        device: GPU device ID (0, 1, etc.) or 'cpu'
        project: Output project directory
        name: Experiment name (auto-generated if None)
        resume: Resume training from last checkpoint
        weights: Pretrained weights to start from
        
    Returns:
        Training results
    """
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("ultralytics package not installed")
    
    # Generate experiment name if not provided
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"defect_yolo11m_{timestamp}"
    
    print(f"\nStarting training: {name}")
    print(f"Dataset config: {data_config}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {img_size}")
    
    # Load pretrained YOLOv11m model
    model = YOLO(weights)
    
    # Training configuration
    results = model.train(
        # Data
        data=data_config,
        
        # Training duration
        epochs=epochs,
        patience=patience,
        
        # Batch and image size
        batch=batch_size,
        imgsz=img_size,
        
        # Hardware
        device=device,
        workers=8,
        
        # Optimization
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Augmentation (tuned for defect detection)
        degrees=15.0,      # Rotation ±15°
        translate=0.1,     # Translation ±10%
        scale=0.3,         # Scale ±30%
        shear=0.0,         # No shear
        perspective=0.0,   # No perspective
        flipud=0.0,        # No vertical flip (uncommon in production)
        fliplr=0.5,        # Horizontal flip 50%
        mosaic=1.0,        # Mosaic augmentation
        mixup=0.1,         # MixUp augmentation
        copy_paste=0.0,    # No copy-paste
        hsv_h=0.015,       # Hue variation
        hsv_s=0.4,         # Saturation variation
        hsv_v=0.4,         # Value (brightness) variation
        
        # Output
        project=project,
        name=name,
        exist_ok=False,
        save=True,
        save_period=-1,    # Save only best and last
        plots=True,
        
        # Validation
        val=True,
        
        # Advanced
        amp=True,          # Automatic Mixed Precision
        resume=resume,
        verbose=True,
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")
    print(f"Last model saved to: {results.save_dir}/weights/last.pt")
    print(f"Training plots saved to: {results.save_dir}")
    print("="*60)
    
    return results


def validate_model(
    model_path: str,
    data_config: str = "config/dataset.yaml",
    img_size: int = 640,
    batch_size: int = 16,
    device: int = 0
):
    """
    Validate trained model and print metrics
    
    Args:
        model_path: Path to trained model weights
        data_config: Path to dataset.yaml
        img_size: Image size for validation
        batch_size: Batch size
        device: GPU device
        
    Returns:
        Validation metrics
    """
    model = YOLO(model_path)
    
    metrics = model.val(
        data=data_config,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        split='test',  # Validate on test set
        plots=True,
        save_json=True,
    )
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    
    print("\nPer-class AP50:")
    for i, class_name in enumerate(model.names.values()):
        print(f"  {class_name}: {metrics.box.ap50[i]:.4f}")
    
    print("="*60)
    
    return metrics


def main():
    """Main training entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv11m for defect detection')
    parser.add_argument('--data', default='config/dataset.yaml',
                       help='Path to dataset.yaml')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--device', default='0',
                       help='GPU device (0, 1, etc.) or cpu')
    parser.add_argument('--weights', default='yolo11m.pt',
                       help='Pretrained weights')
    parser.add_argument('--name', default=None,
                       help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    parser.add_argument('--validate-only', type=str, default=None,
                       help='Only validate model at given path')
    
    args = parser.parse_args()
    
    # Check environment
    has_gpu = check_environment()
    
    # Parse device
    if args.device.lower() == 'cpu':
        device = 'cpu'
    else:
        device = int(args.device)
    
    # Validate only mode
    if args.validate_only:
        validate_model(
            model_path=args.validate_only,
            data_config=args.data,
            img_size=args.imgsz,
            batch_size=args.batch,
            device=device
        )
        return
    
    # Training mode
    train_yolov11m(
        data_config=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        patience=args.patience,
        device=device,
        weights=args.weights,
        name=args.name,
        resume=args.resume
    )


if __name__ == "__main__":
    main()

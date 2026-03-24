"""
YOLOv11m Training Script for Defect Detection
Reads hyperparameters from config/hyperparams.yaml so all tuning
is done in one place.
"""

import os
import sys
import yaml
from pathlib import Path
from datetime import datetime

import torch

# Project root = parent of 'scripts/' directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HYPERPARAMS = str(PROJECT_ROOT / "config" / "hyperparams.yaml")
DEFAULT_DATASET_YAML = str(PROJECT_ROOT / "config" / "dataset.yaml")

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


def load_hyperparams(yaml_path: str, preset: str = "good_vs_rust_optimized") -> dict:
    """
    Load a hyperparameter preset from hyperparams.yaml.

    Args:
        yaml_path: Path to hyperparams.yaml
        preset: Preset name (baseline, small_dataset, fine_detail, fast_training)

    Returns:
        Dictionary of hyperparameter key-value pairs
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        print(f"ERROR: Hyperparams file not found: {yaml_path}")
        sys.exit(1)

    with open(yaml_path, 'r') as f:
        all_presets = yaml.safe_load(f)

    if preset not in all_presets:
        available = [k for k in all_presets if isinstance(all_presets[k], dict)]
        print(f"ERROR: Preset '{preset}' not found. Available: {available}")
        sys.exit(1)

    params = all_presets[preset]
    print(f"Loaded hyperparameter preset: '{preset}'")
    if 'description' in params:
        print(f"  → {params['description']}")
    return params


def train_yolo_model(
    data_config: str = "config/dataset.yaml",
    hyperparams: dict = None,
    device = 0,
    project: str = "models",
    name: str = None,
    resume: bool = False,
    weights: str = "yolo26m.pt"
):
    """
    Train YOLO model on defect detection dataset.

    All training / augmentation / optimizer settings are read from the
    hyperparams dict (loaded from config/hyperparams.yaml).  CLI arguments
    can override individual values.

    Args:
        data_config: Path to dataset.yaml configuration
        hyperparams: Dict of hyperparameters (from load_hyperparams)
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

    if hyperparams is None:
        hyperparams = {}

    # ---- Read values from hyperparams (with sensible fallbacks) ----
    epochs        = hyperparams.get('epochs', 100)
    patience      = hyperparams.get('patience', 20)
    batch_size    = hyperparams.get('batch_size', 4)
    img_size      = hyperparams.get('img_size', 1024)
    optimizer     = hyperparams.get('optimizer', 'auto')
    lr0           = hyperparams.get('lr0', 0.01)
    lrf           = hyperparams.get('lrf', 0.01)
    momentum      = hyperparams.get('momentum', 0.937)
    weight_decay  = hyperparams.get('weight_decay', 0.0005)
    warmup_epochs = hyperparams.get('warmup_epochs', 3.0)
    warmup_momentum = hyperparams.get('warmup_momentum', 0.8)
    warmup_bias_lr  = hyperparams.get('warmup_bias_lr', 0.1)
    workers       = hyperparams.get('workers', 8)
    amp           = hyperparams.get('amp', True)

    # Loss weights
    cls_weight    = hyperparams.get('cls', 0.5)
    box_weight    = hyperparams.get('box', 7.5)
    dfl_weight    = hyperparams.get('dfl', 1.5)

    # Transfer learning
    freeze        = hyperparams.get('freeze', None)
    cos_lr        = hyperparams.get('cos_lr', True)

    # Mosaic control
    close_mosaic  = hyperparams.get('close_mosaic', 10)

    # Augmentation
    degrees     = hyperparams.get('degrees', 0.0)
    translate   = hyperparams.get('translate', 0.1)
    scale       = hyperparams.get('scale', 0.5)
    shear       = hyperparams.get('shear', 0.0)
    perspective = hyperparams.get('perspective', 0.0)
    flipud      = hyperparams.get('flipud', 0.0)
    fliplr      = hyperparams.get('fliplr', 0.5)
    mosaic      = hyperparams.get('mosaic', 1.0)
    mixup       = hyperparams.get('mixup', 0.0)
    copy_paste  = hyperparams.get('copy_paste', 0.0)
    hsv_h       = hyperparams.get('hsv_h', 0.015)
    hsv_s       = hyperparams.get('hsv_s', 0.2)
    hsv_v       = hyperparams.get('hsv_v', 0.4)

    # Generate experiment name if not provided
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"defect_yolo26m_{timestamp}"

    print(f"\nStarting training: {name}")
    print(f"Dataset config: {data_config}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {img_size}")
    print(f"Optimizer: {optimizer}, LR0: {lr0}, Weight Decay: {weight_decay}")
    print(f"Loss weights: cls={cls_weight}, box={box_weight}, dfl={dfl_weight}")
    print(f"Freeze: {freeze}, Close mosaic: {close_mosaic}")

    # Load pretrained model
    model = YOLO(weights)

    # Training — every value comes from hyperparams.yaml
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
        workers=workers,

        # Optimization
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        warmup_momentum=warmup_momentum,
        warmup_bias_lr=warmup_bias_lr,

        # Augmentation
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        flipud=flipud,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,

        # Loss weights
        cls=cls_weight,
        box=box_weight,
        dfl=dfl_weight,

        # Transfer learning
        freeze=freeze,
        cos_lr=cos_lr,

        # Mosaic control
        close_mosaic=close_mosaic,

        # Output
        project=project,
        name=name,
        exist_ok=False,
        save=True,
        save_period=-1,
        plots=True,

        # Validation
        val=True,

        # Advanced
        amp=amp,
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
    img_size: int = 1024,
    batch_size: int = 4,
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

    parser = argparse.ArgumentParser(
        description='Train YOLO for defect detection (reads config/hyperparams.yaml)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train using the fine_detail preset (default)
  python train_model.py

  # Train using a different preset
  python train_model.py --preset small_dataset

  # Override a single value from the preset
  python train_model.py --preset fine_detail --epochs 200

  # Use a custom hyperparams file
  python train_model.py --hyperparams path/to/my_hyperparams.yaml --preset baseline
        """
    )

    # Hyperparams source
    parser.add_argument('--hyperparams', default=DEFAULT_HYPERPARAMS,
                       help='Path to hyperparams YAML file (default: config/hyperparams.yaml)')
    parser.add_argument('--preset', default='good_vs_rust_optimized',
                       help='Preset name inside hyperparams.yaml (default: good_vs_rust_optimized)')

    # Overrides — these take priority over the YAML values
    parser.add_argument('--data', default=DEFAULT_DATASET_YAML,
                       help='Path to dataset.yaml')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override epochs from YAML')
    parser.add_argument('--batch', type=float, default=None,
                       help='Override batch_size (int for fixed, 0.0-1.0 for GPU memory fraction)')
    parser.add_argument('--imgsz', type=int, default=None,
                       help='Override image size')
    parser.add_argument('--patience', type=int, default=None,
                       help='Override early stopping patience')
    parser.add_argument('--device', default='0',
                       help='GPU device (0, 1, etc.) or cpu')
    parser.add_argument('--weights', default=None,
                       help='Override pretrained weights')
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
            img_size=args.imgsz or 1024,
            batch_size=int(args.batch) if args.batch and args.batch >= 1 else (args.batch or 4),
            device=device
        )
        return

    # ---- Load hyperparams from YAML ----
    hp = load_hyperparams(args.hyperparams, args.preset)

    # ---- Apply CLI overrides (only if explicitly provided) ----
    if args.epochs is not None:
        hp['epochs'] = args.epochs
    if args.batch is not None:
        hp['batch_size'] = int(args.batch) if args.batch >= 1 else args.batch
    if args.imgsz is not None:
        hp['img_size'] = args.imgsz
    if args.patience is not None:
        hp['patience'] = args.patience

    # ---- Determine weights ----
    weights = args.weights if args.weights else 'yolo26m.pt'

    # ---- Train ----
    train_yolo_model(
        data_config=args.data,
        hyperparams=hp,
        device=device,
        weights=weights,
        name=args.name,
        resume=args.resume
    )


if __name__ == "__main__":
    main()

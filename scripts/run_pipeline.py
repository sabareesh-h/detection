"""
Master Pipeline Script — Defect Detection System
Chains all pipeline steps into a single command:
  Data Preparation → Augmentation → Training → Evaluation → Export

Usage:
  python scripts/run_pipeline.py --mode full          # Run everything
  python scripts/run_pipeline.py --mode train-eval    # Train + Evaluate only
  python scripts/run_pipeline.py --mode eval-only     # Evaluate existing model
  python scripts/run_pipeline.py --mode augment-only  # Augment dataset only
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default paths
DEFAULT_DATA_CONFIG = str(PROJECT_ROOT / "config" / "dataset.yaml")
DEFAULT_HYPERPARAMS = str(PROJECT_ROOT / "config" / "hyperparams.yaml")
DEFAULT_TRAIN_IMAGES = str(PROJECT_ROOT / "dataset" / "images" / "train")
DEFAULT_TRAIN_LABELS = str(PROJECT_ROOT / "dataset" / "labels" / "train")
DEFAULT_MODEL_WEIGHTS = "yolo11m.pt"


# ============================================================
# HELPER: Load hyperparameter preset
# ============================================================

def load_preset(preset_name: str) -> dict:
    """Load a hyperparameter preset from config/hyperparams.yaml."""
    try:
        import yaml
    except ImportError:
        print("WARNING: pyyaml not installed, using default values.")
        return {}

    hyperparams_path = PROJECT_ROOT / "config" / "hyperparams.yaml"
    if not hyperparams_path.exists():
        print(f"WARNING: {hyperparams_path} not found, using defaults.")
        return {}

    with open(hyperparams_path, 'r') as f:
        all_presets = yaml.safe_load(f)

    if preset_name not in all_presets:
        available = [k for k in all_presets.keys() if isinstance(all_presets[k], dict)]
        print(f"WARNING: Preset '{preset_name}' not found. Available: {available}")
        return {}

    preset = all_presets[preset_name]
    print(f"  Loaded preset: {preset_name}")
    print(f"  Description: {preset.get('description', 'N/A')}")
    return preset


# ============================================================
# PIPELINE STEPS
# ============================================================

def step_validate(images_dir: str) -> bool:
    """Step 1: Validate image quality."""
    print_step_header("1. VALIDATE IMAGES")

    from validate_images import ImageQualityValidator
    validator = ImageQualityValidator()
    results = validator.print_report(images_dir)

    if results['total'] == 0:
        print("ERROR: No images found. Cannot proceed.")
        return False

    fail_rate = results['failed'] / max(results['total'], 1)
    if fail_rate > 0.5:
        print(f"WARNING: {fail_rate*100:.0f}% of images failed quality checks.")
        print("Consider re-capturing or adjusting quality thresholds.")

    return True


def step_augment(images_dir: str, labels_dir: str, multiplier: int = 5,
                 level: str = "medium") -> bool:
    """Step 2: Augment training data."""
    print_step_header("2. AUGMENT TRAINING DATA")

    try:
        from augment_dataset import augment_dataset
    except ImportError:
        print("ERROR: augment_dataset.py not found or albumentations not installed.")
        print("Install with: pip install albumentations>=1.3.0")
        return False

    # Create a separate output directory to keep originals untouched
    output_images = str(Path(images_dir).parent / "train_augmented")
    output_labels = str(Path(labels_dir).parent / "train_augmented")

    result = augment_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_images_dir=output_images,
        output_labels_dir=output_labels,
        multiplier=multiplier,
        level=level,
        copy_originals=True
    )

    if result['total_augmented'] == 0:
        print("WARNING: No augmented images generated.")
        return False

    print(f"\nAugmented dataset ready at:")
    print(f"  Images: {output_images}")
    print(f"  Labels: {output_labels}")
    return True


def step_train(data_config: str, preset_name: str = "small_dataset",
               weights: str = DEFAULT_MODEL_WEIGHTS,
               device: str = "0", name: str = None) -> str:
    """Step 3: Train the model. Returns path to best weights."""
    print_step_header("3. TRAIN MODEL")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.")
        return None

    import torch
    from train_model import check_environment, train_yolov11m

    # Check GPU
    check_environment()

    # Load preset
    preset = load_preset(preset_name)

    # Generate experiment name
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"pipeline_{preset_name}_{timestamp}"

    # Train with preset values
    results = train_yolov11m(
        data_config=data_config,
        epochs=preset.get('epochs', 100),
        batch_size=preset.get('batch_size', 16),
        img_size=preset.get('img_size', 640),
        patience=preset.get('patience', 20),
        device=int(device) if device != 'cpu' else device,
        weights=weights,
        name=name,
    )

    best_weights = str(Path(results.save_dir) / "weights" / "best.pt")
    print(f"\nBest model saved: {best_weights}")
    return best_weights


def step_evaluate(model_path: str, data_config: str, device: str = "0",
                  split: str = "test") -> dict:
    """Step 4: Evaluate the model."""
    print_step_header("4. EVALUATE MODEL")

    from evaluate_model import ModelEvaluator

    evaluator = ModelEvaluator(
        model_path=model_path,
        data_config=data_config,
        device=device,
    )

    results = evaluator.run_full_evaluation(
        split=split,
        conf_threshold=0.5,
        benchmark_iterations=50,
    )

    return results


def step_compare() -> None:
    """Step 5: Compare all training runs."""
    print_step_header("5. COMPARE TRAINING RUNS")

    from compare_runs import discover_runs, print_comparison_table, plot_comparison

    all_runs = []
    for run_dir in ['runs/detect', 'models']:
        full_path = str(PROJECT_ROOT / run_dir)
        runs = discover_runs(full_path)
        all_runs.extend(runs)

    if all_runs:
        # De-duplicate
        seen = set()
        unique = [r for r in all_runs if r['path'] not in seen and not seen.add(r['path'])]
        print_comparison_table(unique)
        plot_comparison(unique, str(PROJECT_ROOT / "logs" / "run_comparison.png"))
    else:
        print("No previous runs found to compare.")


def step_export(model_path: str, fmt: str = "onnx") -> bool:
    """Step 6: Export model for production."""
    print_step_header("6. EXPORT MODEL")

    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        model.export(format=fmt)
        print(f"Model exported to {fmt} format.")
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False


# ============================================================
# PIPELINE MODES
# ============================================================

def run_full_pipeline(args):
    """Full pipeline: validate → augment → train → evaluate → compare."""
    start_time = time.time()

    print_pipeline_header("FULL PIPELINE")

    # Step 1: Validate
    if not step_validate(args.train_images):
        print("\nPipeline stopped at validation.")
        return

    # Step 2: Augment
    step_augment(
        images_dir=args.train_images,
        labels_dir=args.train_labels,
        multiplier=args.multiplier,
        level=args.aug_level,
    )

    # Step 3: Train
    best_model = step_train(
        data_config=args.data,
        preset_name=args.preset,
        weights=args.weights,
        device=args.device,
        name=args.name,
    )

    if best_model is None or not os.path.exists(best_model):
        print("\nPipeline stopped: training failed.")
        return

    # Step 4: Evaluate
    eval_results = step_evaluate(best_model, args.data, args.device)

    # Step 5: Compare
    step_compare()

    # Step 6: Export (optional)
    if args.export:
        step_export(best_model, args.export_format)

    elapsed = time.time() - start_time
    print_pipeline_footer(elapsed, best_model, eval_results)


def run_train_eval(args):
    """Train + Evaluate only (skip augmentation)."""
    start_time = time.time()
    print_pipeline_header("TRAIN + EVALUATE")

    best_model = step_train(
        data_config=args.data,
        preset_name=args.preset,
        weights=args.weights,
        device=args.device,
        name=args.name,
    )

    if best_model and os.path.exists(best_model):
        eval_results = step_evaluate(best_model, args.data, args.device)
        step_compare()
        print_pipeline_footer(time.time() - start_time, best_model, eval_results)
    else:
        print("Training failed.")


def run_eval_only(args):
    """Evaluate an existing model."""
    print_pipeline_header("EVALUATE ONLY")

    if not args.model:
        print("ERROR: --model is required for eval-only mode.")
        return

    eval_results = step_evaluate(args.model, args.data, args.device)
    step_compare()


def run_augment_only(args):
    """Augment dataset only."""
    print_pipeline_header("AUGMENT ONLY")

    step_validate(args.train_images)
    step_augment(
        images_dir=args.train_images,
        labels_dir=args.train_labels,
        multiplier=args.multiplier,
        level=args.aug_level,
    )


# ============================================================
# FORMATTING HELPERS
# ============================================================

def print_step_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_pipeline_header(mode: str):
    print(f"\n{'#'*60}")
    print(f"#  DEFECT DETECTION PIPELINE — {mode}")
    print(f"#  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")


def print_pipeline_footer(elapsed: float, model_path: str = None, eval_results: dict = None):
    mins, secs = divmod(int(elapsed), 60)
    hours, mins = divmod(mins, 60)

    print(f"\n{'#'*60}")
    print(f"#  PIPELINE COMPLETE")
    print(f"#  Duration: {hours}h {mins}m {secs}s")
    if model_path:
        print(f"#  Best model: {model_path}")
    if eval_results and 'metrics' in eval_results:
        overall = eval_results['metrics'].get('overall', {})
        print(f"#  mAP@50:    {overall.get('mAP50', 'N/A')}")
        print(f"#  mAP@50-95: {overall.get('mAP50_95', 'N/A')}")
        print(f"#  F1 Score:  {overall.get('f1', 'N/A')}")
    print(f"{'#'*60}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Master Pipeline — Defect Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Modes:
  full          Validate → Augment → Train → Evaluate → Compare → Export
  train-eval    Train → Evaluate → Compare (skip augmentation)
  eval-only     Evaluate an existing model (requires --model)
  augment-only  Augment training dataset only

Hyperparameter Presets (from config/hyperparams.yaml):
  baseline       Default training config
  small_dataset  Tuned for <200 images (recommended for your current dataset)
  fine_detail    High-res training for small/subtle defects (imgsz=1280)
  fast_training  Quick sanity check (20 epochs)

Examples:
  # Full pipeline with small_dataset preset (RECOMMENDED)
  python scripts/run_pipeline.py --mode full --preset small_dataset

  # Quick test to verify everything works
  python scripts/run_pipeline.py --mode full --preset fast_training

  # Train + evaluate with fine detail preset
  python scripts/run_pipeline.py --mode train-eval --preset fine_detail

  # Evaluate an existing model
  python scripts/run_pipeline.py --mode eval-only --model models/best.pt

  # Augment dataset with heavy augmentation (10x)
  python scripts/run_pipeline.py --mode augment-only --multiplier 10 --aug-level heavy
        """
    )

    parser.add_argument('--mode', required=True,
                        choices=['full', 'train-eval', 'eval-only', 'augment-only'],
                        help='Pipeline mode to run')
    parser.add_argument('--preset', default='small_dataset',
                        choices=['baseline', 'small_dataset', 'fine_detail', 'fast_training'],
                        help='Hyperparameter preset (default: small_dataset)')
    parser.add_argument('--data', default=DEFAULT_DATA_CONFIG,
                        help='Path to dataset.yaml')
    parser.add_argument('--weights', default=DEFAULT_MODEL_WEIGHTS,
                        help='Pre-trained model weights (default: yolo11m.pt)')
    parser.add_argument('--model', default=None,
                        help='Path to trained model (for eval-only mode)')
    parser.add_argument('--device', default='0',
                        help='Device — GPU id or "cpu"')
    parser.add_argument('--name', default=None,
                        help='Experiment name (auto-generated if not set)')

    # Augmentation options
    parser.add_argument('--train-images', default=DEFAULT_TRAIN_IMAGES,
                        help='Training images directory')
    parser.add_argument('--train-labels', default=DEFAULT_TRAIN_LABELS,
                        help='Training labels directory')
    parser.add_argument('--multiplier', type=int, default=5,
                        help='Augmentation multiplier (default: 5)')
    parser.add_argument('--aug-level', default='medium',
                        choices=['light', 'medium', 'heavy'],
                        help='Augmentation intensity (default: medium)')

    # Export options
    parser.add_argument('--export', action='store_true',
                        help='Export model after training')
    parser.add_argument('--export-format', default='onnx',
                        help='Export format: onnx, torchscript, etc.')

    args = parser.parse_args()

    # Add scripts dir to path for imports
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    # Dispatch to pipeline mode
    mode_map = {
        'full': run_full_pipeline,
        'train-eval': run_train_eval,
        'eval-only': run_eval_only,
        'augment-only': run_augment_only,
    }

    mode_map[args.mode](args)


if __name__ == "__main__":
    main()

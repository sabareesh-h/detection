"""
Model Evaluation Report Script for Defect Detection
Generates a comprehensive evaluation report: confusion matrix, per-class metrics,
mAP, confidence histograms, inference speed benchmarks, and saves everything
to a timestamped report directory.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Error: ultralytics not installed. Run: pip install ultralytics")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not installed. Plots will be skipped.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ============================================================
# EVALUATION ENGINE
# ============================================================

class ModelEvaluator:
    """
    Comprehensive YOLO model evaluation — generates full report
    with metrics, confusion matrix, speed benchmarks, and confidence analysis.
    """

    def __init__(self, model_path: str, data_config: str = "config/dataset.yaml",
                 device: str = "0", output_dir: str = None):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained YOLO model (.pt file)
            data_config: Path to dataset.yaml
            device: Device to run on ('0' for GPU, 'cpu' for CPU)
            output_dir: Output directory for report (auto-generated if None)
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics is required for evaluation")

        self.model_path = model_path
        self.data_config = data_config
        self.device = device

        # Load model
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names  # {0: 'Good', 1: 'Flat_line', 2: 'Unwash'}

        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"logs/evaluation_report_{timestamp}"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Output directory: {self.output_dir}")
        print(f"Classes: {self.class_names}")
        print(f"Device: {device}")

    def run_full_evaluation(self, split: str = "test", img_size: int = 640,
                            batch_size: int = 16, conf_threshold: float = 0.5,
                            iou_threshold: float = 0.45,
                            benchmark_iterations: int = 100) -> Dict:
        """
        Run complete evaluation pipeline.
        
        Args:
            split: Dataset split to evaluate on ('test', 'val')
            img_size: Input image size
            batch_size: Batch size for evaluation
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            benchmark_iterations: Number of iterations for speed benchmark
            
        Returns:
            Complete evaluation results dict
        """
        print(f"\n{'='*60}")
        print(f"FULL MODEL EVALUATION")
        print(f"{'='*60}")
        print(f"Model:      {self.model_path}")
        print(f"Dataset:    {self.data_config}")
        print(f"Split:      {split}")
        print(f"Image size: {img_size}")
        print(f"Conf:       {conf_threshold}")
        print(f"IoU:        {iou_threshold}")
        print(f"{'='*60}\n")

        results = {
            'model_path': self.model_path,
            'data_config': self.data_config,
            'device': self.device,
            'split': split,
            'img_size': img_size,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'timestamp': datetime.now().isoformat(),
            'class_names': {str(k): v for k, v in self.class_names.items()},
        }

        # Step 1: Ultralytics validation (mAP, precision, recall)
        print("Step 1/4: Running YOLO validation (mAP, P, R)...")
        val_metrics = self._run_validation(split, img_size, batch_size, conf_threshold, iou_threshold)
        results['metrics'] = val_metrics

        # Step 2: Confusion matrix
        print("\nStep 2/4: Generating confusion matrix...")
        self._generate_confusion_matrix(results)

        # Step 3: Confidence histogram
        print("\nStep 3/4: Generating confidence histogram...")
        self._generate_confidence_histogram(split, img_size, conf_threshold)

        # Step 4: Speed benchmark
        print("\nStep 4/4: Running speed benchmark...")
        speed_results = self._benchmark_speed(img_size, benchmark_iterations)
        results['speed'] = speed_results

        # Save report
        self._save_report(results)
        self._print_summary(results)

        return results

    # --------------------------------------------------------
    # Step 1: Validation Metrics
    # --------------------------------------------------------

    def _run_validation(self, split: str, img_size: int, batch_size: int,
                        conf: float, iou: float) -> Dict:
        """Run Ultralytics validation and extract metrics."""
        metrics = self.model.val(
            data=self.data_config,
            imgsz=img_size,
            batch=batch_size,
            device=self.device,
            split=split,
            conf=conf,
            iou=iou,
            plots=True,
            save_json=True,
            project=self.output_dir,
            name='val_results',
            exist_ok=True,
        )

        # Extract overall metrics
        val_metrics = {
            'overall': {
                'mAP50': round(float(metrics.box.map50), 4),
                'mAP50_95': round(float(metrics.box.map), 4),
                'precision': round(float(metrics.box.mp), 4),
                'recall': round(float(metrics.box.mr), 4),
                'f1': round(2 * float(metrics.box.mp) * float(metrics.box.mr) /
                           max(float(metrics.box.mp) + float(metrics.box.mr), 1e-6), 4),
            },
            'per_class': {}
        }

        # Per-class metrics
        for i, class_name in self.class_names.items():
            try:
                class_metrics = {
                    'ap50': round(float(metrics.box.ap50[i]), 4),
                    'ap50_95': round(float(metrics.box.ap[i]), 4),
                    'precision': round(float(metrics.box.p[i]), 4) if hasattr(metrics.box, 'p') else None,
                    'recall': round(float(metrics.box.r[i]), 4) if hasattr(metrics.box, 'r') else None,
                }
                # Calculate per-class F1
                p = class_metrics.get('precision') or 0
                r = class_metrics.get('recall') or 0
                class_metrics['f1'] = round(2 * p * r / max(p + r, 1e-6), 4)
                val_metrics['per_class'][class_name] = class_metrics
            except (IndexError, AttributeError) as e:
                print(f"  WARNING: Could not extract metrics for class {class_name}: {e}")

        return val_metrics

    # --------------------------------------------------------
    # Step 2: Confusion Matrix
    # --------------------------------------------------------

    def _generate_confusion_matrix(self, results: Dict):
        """Generate and save confusion matrix plot."""
        if not MATPLOTLIB_AVAILABLE:
            print("  Skipping — matplotlib not installed.")
            return

        # Ultralytics saves confusion_matrix.png in val results
        # Check if it already exists from the val() call
        val_results_dir = os.path.join(self.output_dir, 'val_results')
        cm_source = os.path.join(val_results_dir, 'confusion_matrix.png')
        cm_norm_source = os.path.join(val_results_dir, 'confusion_matrix_normalized.png')

        if os.path.exists(cm_source):
            print(f"  Confusion matrix saved by Ultralytics: {cm_source}")
        if os.path.exists(cm_norm_source):
            print(f"  Normalized confusion matrix: {cm_norm_source}")

        # Note: Per-class confusion details are included in the val_metrics
        # The Ultralytics val() already generates the confusion matrix plots.

    # --------------------------------------------------------
    # Step 3: Confidence Histogram
    # --------------------------------------------------------

    def _generate_confidence_histogram(self, split: str, img_size: int, conf: float):
        """
        Run inference on the dataset and plot confidence score distribution.
        Helps determine optimal confidence threshold.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("  Skipping — matplotlib not installed.")
            return

        import yaml

        # Read dataset config
        with open(self.data_config, 'r') as f:
            data_cfg = yaml.safe_load(f)

        dataset_path = data_cfg.get('path', '')
        split_subdir = data_cfg.get(split, data_cfg.get('test', data_cfg.get('val', 'images/test')))
        images_dir = os.path.join(dataset_path, split_subdir)

        if not os.path.exists(images_dir):
            print(f"  WARNING: Images directory not found: {images_dir}")
            return

        # Collect all confidence scores
        all_confidences = {name: [] for name in self.class_names.values()}

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            import glob 
            image_files.extend(glob.glob(os.path.join(images_dir, ext)))

        print(f"  Running inference on {len(image_files)} images...")

        for img_path in image_files:
            results = self.model.predict(
                img_path, imgsz=img_size, conf=0.01,  # Very low conf to capture full distribution
                device=self.device, verbose=False
            )
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.class_names.get(cls_id, f"Unknown_{cls_id}")
                        if class_name in all_confidences:
                            all_confidences[class_name].append(confidence)

        # Plot histogram
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Combined histogram
        ax1 = axes[0]
        all_confs = []
        for name, confs in all_confidences.items():
            all_confs.extend(confs)
            if confs:
                ax1.hist(confs, bins=50, alpha=0.5, label=f"{name} (n={len(confs)})")
        ax1.axvline(x=conf, color='red', linestyle='--', linewidth=2, label=f"Threshold ({conf})")
        ax1.set_title('Confidence Score Distribution by Class', fontsize=12)
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Count')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Per-class box plot
        ax2 = axes[1]
        class_data = []
        class_labels = []
        for name, confs in all_confidences.items():
            if confs:
                class_data.append(confs)
                class_labels.append(f"{name}\n(n={len(confs)})")
        if class_data:
            bp = ax2.boxplot(class_data, labels=class_labels, patch_artist=True)
            colors_box = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db', '#9b59b6']
            for patch, color in zip(bp['boxes'], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax2.axhline(y=conf, color='red', linestyle='--', linewidth=2, label=f"Threshold ({conf})")
            ax2.set_title('Confidence Distribution per Class', fontsize=12)
            ax2.set_ylabel('Confidence Score')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        hist_path = os.path.join(self.output_dir, 'confidence_histogram.png')
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Confidence histogram saved: {hist_path}")

        # Print threshold analysis
        if all_confs:
            print(f"\n  Confidence Analysis:")
            for threshold in [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                above = sum(1 for c in all_confs if c >= threshold)
                pct = above / len(all_confs) * 100
                marker = " ◄ current" if abs(threshold - conf) < 0.01 else ""
                print(f"    conf≥{threshold:.2f}: {above:>5} detections ({pct:>5.1f}%){marker}")

    # --------------------------------------------------------
    # Step 4: Speed Benchmark
    # --------------------------------------------------------

    def _benchmark_speed(self, img_size: int, iterations: int = 100) -> Dict:
        """
        Benchmark inference speed on the current hardware.
        
        Returns:
            Dict with preprocess, inference, postprocess times and FPS
        """
        # Create a dummy image for benchmarking
        dummy_image = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

        # Warmup
        print(f"  Warming up ({min(10, iterations)} iterations)...")
        for _ in range(min(10, iterations)):
            self.model.predict(dummy_image, imgsz=img_size, device=self.device, verbose=False)

        # Benchmark
        print(f"  Benchmarking ({iterations} iterations)...")
        times = []
        preprocess_times = []
        inference_times = []
        postprocess_times = []

        for _ in range(iterations):
            start = time.perf_counter()
            results = self.model.predict(dummy_image, imgsz=img_size, device=self.device, verbose=False)
            end = time.perf_counter()
            total_ms = (end - start) * 1000
            times.append(total_ms)

            # Extract Ultralytics speed breakdown if available
            if results and hasattr(results[0], 'speed'):
                speed = results[0].speed
                preprocess_times.append(speed.get('preprocess', 0))
                inference_times.append(speed.get('inference', 0))
                postprocess_times.append(speed.get('postprocess', 0))

        speed_results = {
            'total_ms_mean': round(np.mean(times), 2),
            'total_ms_std': round(np.std(times), 2),
            'total_ms_min': round(np.min(times), 2),
            'total_ms_max': round(np.max(times), 2),
            'fps': round(1000 / np.mean(times), 1),
            'iterations': iterations,
            'img_size': img_size,
        }

        if preprocess_times:
            speed_results['preprocess_ms'] = round(np.mean(preprocess_times), 2)
            speed_results['inference_ms'] = round(np.mean(inference_times), 2)
            speed_results['postprocess_ms'] = round(np.mean(postprocess_times), 2)

        # Hardware info
        if TORCH_AVAILABLE:
            speed_results['device_name'] = (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() and self.device != 'cpu'
                else 'CPU'
            )
            if torch.cuda.is_available():
                speed_results['gpu_memory_gb'] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 1
                )

        return speed_results

    # --------------------------------------------------------
    # Report Generation
    # --------------------------------------------------------

    def _save_report(self, results: Dict):
        """Save evaluation report as JSON."""
        report_path = os.path.join(self.output_dir, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nReport JSON saved: {report_path}")

        # Also save a human-readable markdown report
        md_path = os.path.join(self.output_dir, 'evaluation_report.md')
        self._save_markdown_report(results, md_path)
        print(f"Report Markdown saved: {md_path}")

    def _save_markdown_report(self, results: Dict, md_path: str):
        """Generate a human-readable Markdown evaluation report."""
        metrics = results.get('metrics', {})
        overall = metrics.get('overall', {})
        per_class = metrics.get('per_class', {})
        speed = results.get('speed', {})

        lines = [
            f"# Model Evaluation Report",
            f"",
            f"**Generated**: {results.get('timestamp', 'N/A')}",
            f"",
            f"## Model Info",
            f"",
            f"| Property | Value |",
            f"|----------|-------|",
            f"| Model | `{results.get('model_path', 'N/A')}` |",
            f"| Dataset | `{results.get('data_config', 'N/A')}` |",
            f"| Split | `{results.get('split', 'N/A')}` |",
            f"| Image Size | `{results.get('img_size', 'N/A')}` |",
            f"| Confidence | `{results.get('conf_threshold', 'N/A')}` |",
            f"| IoU | `{results.get('iou_threshold', 'N/A')}` |",
            f"| Device | `{speed.get('device_name', results.get('device', 'N/A'))}` |",
            f"",
            f"## Overall Metrics",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| mAP@50 | **{overall.get('mAP50', 'N/A')}** |",
            f"| mAP@50-95 | **{overall.get('mAP50_95', 'N/A')}** |",
            f"| Precision | {overall.get('precision', 'N/A')} |",
            f"| Recall | {overall.get('recall', 'N/A')} |",
            f"| F1 Score | {overall.get('f1', 'N/A')} |",
            f"",
            f"## Per-Class Metrics",
            f"",
            f"| Class | AP@50 | AP@50-95 | Precision | Recall | F1 |",
            f"|-------|-------|----------|-----------|--------|-----|",
        ]

        for class_name, class_metrics in per_class.items():
            lines.append(
                f"| {class_name} | {class_metrics.get('ap50', 'N/A')} | "
                f"{class_metrics.get('ap50_95', 'N/A')} | "
                f"{class_metrics.get('precision', 'N/A')} | "
                f"{class_metrics.get('recall', 'N/A')} | "
                f"{class_metrics.get('f1', 'N/A')} |"
            )

        lines.extend([
            f"",
            f"## Inference Speed",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| FPS | **{speed.get('fps', 'N/A')}** |",
            f"| Total (mean) | {speed.get('total_ms_mean', 'N/A')} ms |",
            f"| Total (std) | ±{speed.get('total_ms_std', 'N/A')} ms |",
            f"| Preprocess | {speed.get('preprocess_ms', 'N/A')} ms |",
            f"| Inference | {speed.get('inference_ms', 'N/A')} ms |",
            f"| Postprocess | {speed.get('postprocess_ms', 'N/A')} ms |",
            f"| Device | {speed.get('device_name', 'N/A')} |",
            f"",
            f"## Generated Files",
            f"",
            f"- `evaluation_report.json` — Full metrics data",
            f"- `confidence_histogram.png` — Confidence score distribution",
            f"- `val_results/confusion_matrix.png` — Confusion matrix",
            f"- `val_results/confusion_matrix_normalized.png` — Normalized confusion matrix",
            f"- `val_results/` — Full Ultralytics validation output",
        ])

        with open(md_path, 'w') as f:
            f.write('\n'.join(lines))

    def _print_summary(self, results: Dict):
        """Print a formatted summary to console."""
        metrics = results.get('metrics', {})
        overall = metrics.get('overall', {})
        per_class = metrics.get('per_class', {})
        speed = results.get('speed', {})

        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"  mAP@50:     {overall.get('mAP50', 'N/A')}")
        print(f"  mAP@50-95:  {overall.get('mAP50_95', 'N/A')}")
        print(f"  Precision:  {overall.get('precision', 'N/A')}")
        print(f"  Recall:     {overall.get('recall', 'N/A')}")
        print(f"  F1 Score:   {overall.get('f1', 'N/A')}")

        if per_class:
            print(f"\n  Per-Class AP@50:")
            for class_name, class_metrics in per_class.items():
                ap50 = class_metrics.get('ap50', 'N/A')
                f1 = class_metrics.get('f1', 'N/A')
                print(f"    {class_name:15s}  AP50={ap50}  F1={f1}")

        print(f"\n  Inference Speed:")
        print(f"    FPS:           {speed.get('fps', 'N/A')}")
        print(f"    Latency:       {speed.get('total_ms_mean', 'N/A')}ms ±{speed.get('total_ms_std', 'N/A')}ms")
        print(f"    Device:        {speed.get('device_name', 'N/A')}")

        print(f"\n  Full report:     {self.output_dir}")
        print(f"{'='*60}")


# ============================================================
# CLI
# ============================================================

def main():
    """CLI entry point for model evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Comprehensive YOLO model evaluation report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate best model on test set
  python evaluate_model.py --model models/best.pt

  # Evaluate on validation set with custom threshold
  python evaluate_model.py --model runs/detect/models/weights/best.pt --split val --conf 0.4

  # CPU-only evaluation with custom output
  python evaluate_model.py --model models/best.pt --device cpu --output logs/my_eval

  # Quick benchmark only (fewer iterations)
  python evaluate_model.py --model models/best.pt --benchmark-iters 50
        """
    )

    parser.add_argument('--model', required=True,
                        help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--data', default='config/dataset.yaml',
                        help='Path to dataset.yaml (default: config/dataset.yaml)')
    parser.add_argument('--split', default='test', choices=['test', 'val', 'train'],
                        help='Dataset split to evaluate (default: test)')
    parser.add_argument('--imgsz', type=int, default=1024,
                        help='Input image size (default: 1024)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--device', default='0',
                        help='Device — GPU id or "cpu" (default: 0)')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: auto-generated with timestamp)')
    parser.add_argument('--benchmark-iters', type=int, default=100,
                        help='Number of iterations for speed benchmark (default: 100)')

    args = parser.parse_args()

    if not ULTRALYTICS_AVAILABLE:
        print("\nERROR: ultralytics is required.")
        print("Install with: pip install ultralytics>=8.3.0")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"\nERROR: Model not found: {args.model}")
        sys.exit(1)

    evaluator = ModelEvaluator(
        model_path=args.model,
        data_config=args.data,
        device=args.device,
        output_dir=args.output,
    )

    evaluator.run_full_evaluation(
        split=args.split,
        img_size=args.imgsz,
        batch_size=args.batch,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        benchmark_iterations=args.benchmark_iters,
    )


if __name__ == "__main__":
    main()

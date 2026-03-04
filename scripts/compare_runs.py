"""
Training Run Comparison Script
Scans Ultralytics training run directories, extracts metrics from results.csv,
and generates comparison tables and overlay plots across experiments.
"""

import os
import sys
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving plots
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not installed. Plots will be skipped.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# ============================================================
# PARSING ULTRALYTICS RESULTS
# ============================================================

# Column name mapping — Ultralytics results.csv uses these headers
# (they may have leading/trailing spaces)
METRIC_COLUMNS = {
    'train/box_loss': 'Train Box Loss',
    'train/cls_loss': 'Train Class Loss',
    'train/dfl_loss': 'Train DFL Loss',
    'val/box_loss': 'Val Box Loss',
    'val/cls_loss': 'Val Class Loss',
    'val/dfl_loss': 'Val DFL Loss',
    'metrics/precision(B)': 'Precision',
    'metrics/recall(B)': 'Recall',
    'metrics/mAP50(B)': 'mAP50',
    'metrics/mAP50-95(B)': 'mAP50-95',
    'lr/pg0': 'LR (pg0)',
    'lr/pg1': 'LR (pg1)',
    'lr/pg2': 'LR (pg2)',
}


def parse_results_csv(csv_path: str) -> Optional[Dict]:
    """
    Parse an Ultralytics results.csv file.
    
    Returns:
        Dict with run name, epoch data, and best metrics
    """
    if not os.path.exists(csv_path):
        return None

    try:
        rows = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Strip whitespace from keys and values
                cleaned = {k.strip(): v.strip() for k, v in row.items()}
                rows.append(cleaned)

        if not rows:
            return None

        # Extract per-epoch metrics
        epochs = []
        for row in rows:
            epoch_data = {'epoch': int(row.get('epoch', 0))}
            for csv_key, display_name in METRIC_COLUMNS.items():
                val = row.get(csv_key, None)
                if val is not None:
                    try:
                        epoch_data[display_name] = float(val)
                    except ValueError:
                        pass
            epochs.append(epoch_data)

        # Find best epoch (by mAP50-95)
        best_epoch = max(epochs, key=lambda e: e.get('mAP50-95', 0))

        # Get final epoch metrics
        final_epoch = epochs[-1] if epochs else {}

        return {
            'epochs': epochs,
            'total_epochs': len(epochs),
            'best': best_epoch,
            'final': final_epoch,
        }

    except Exception as e:
        print(f"  ERROR parsing {csv_path}: {e}")
        return None


def discover_runs(runs_dir: str) -> List[Dict]:
    """
    Discover all training runs under a directory.
    Looks for results.csv files in any subdirectory.
    
    Returns:
        List of dicts with run name, path, and parsed metrics
    """
    runs = []
    runs_path = Path(runs_dir)

    if not runs_path.exists():
        print(f"WARNING: Runs directory not found: {runs_dir}")
        return runs

    # Search for results.csv files recursively
    for csv_file in runs_path.rglob("results.csv"):
        run_dir = csv_file.parent
        run_name = run_dir.name

        # Parse the results
        data = parse_results_csv(str(csv_file))
        if data is not None:
            # Get run metadata
            mod_time = os.path.getmtime(str(csv_file))
            run_info = {
                'name': run_name,
                'path': str(run_dir),
                'csv_path': str(csv_file),
                'modified': datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M'),
                'data': data,
            }

            # Check for args.yaml to get training config
            args_yaml = run_dir / 'args.yaml'
            if args_yaml.exists():
                try:
                    import yaml
                    with open(args_yaml, 'r') as f:
                        run_info['args'] = yaml.safe_load(f)
                except Exception:
                    pass

            runs.append(run_info)

    # Sort by modification time (most recent first)
    runs.sort(key=lambda r: r['modified'], reverse=True)
    return runs


# ============================================================
# COMPARISON TABLE
# ============================================================

def print_comparison_table(runs: List[Dict]):
    """Print a formatted comparison table of all discovered runs."""
    if not runs:
        print("No training runs found.")
        return

    print(f"\n{'='*100}")
    print(f"TRAINING RUN COMPARISON — {len(runs)} run(s) found")
    print(f"{'='*100}")

    # Header
    header = f"{'Run Name':<35} {'Epochs':>7} {'mAP50':>8} {'mAP50-95':>9} {'Prec':>7} {'Recall':>7} {'Date':<16}"
    print(header)
    print("-" * 100)

    for run in runs:
        best = run['data']['best']
        name = run['name'][:34]
        epochs = run['data']['total_epochs']
        map50 = best.get('mAP50', 0)
        map5095 = best.get('mAP50-95', 0)
        precision = best.get('Precision', 0)
        recall = best.get('Recall', 0)
        date = run['modified']

        print(f"{name:<35} {epochs:>7} {map50:>8.4f} {map5095:>9.4f} {precision:>7.4f} {recall:>7.4f} {date:<16}")

    # Best run
    if len(runs) > 1:
        best_run = max(runs, key=lambda r: r['data']['best'].get('mAP50-95', 0))
        print("-" * 100)
        print(f"🏆 Best run: {best_run['name']} (mAP50-95: {best_run['data']['best'].get('mAP50-95', 0):.4f})")

    print(f"{'='*100}\n")


def print_detailed_run(run: Dict):
    """Print detailed info for a single run."""
    print(f"\n--- {run['name']} ---")
    print(f"  Path:   {run['path']}")
    print(f"  Date:   {run['modified']}")
    print(f"  Epochs: {run['data']['total_epochs']}")

    best = run['data']['best']
    print(f"\n  Best Epoch ({best.get('epoch', '?')}):")
    for key in ['mAP50', 'mAP50-95', 'Precision', 'Recall',
                'Val Box Loss', 'Val Class Loss']:
        if key in best:
            print(f"    {key}: {best[key]:.4f}")

    # Show training args if available
    if 'args' in run:
        args = run['args']
        print(f"\n  Training Config:")
        for key in ['imgsz', 'batch', 'epochs', 'lr0', 'optimizer',
                     'mosaic', 'mixup', 'degrees', 'patience']:
            if key in args:
                print(f"    {key}: {args[key]}")


# ============================================================
# PLOTTING
# ============================================================

def plot_comparison(runs: List[Dict], output_path: str = "logs/run_comparison.png"):
    """
    Generate overlay plots comparing metrics across runs.
    
    Plots: mAP50, mAP50-95, Precision, Recall, Val Box Loss, Train Box Loss
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping plots — matplotlib not installed.")
        return

    if not runs:
        print("No runs to plot.")
        return

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    metrics_to_plot = [
        ('mAP50', 'mAP@50'),
        ('mAP50-95', 'mAP@50-95'),
        ('Precision', 'Precision'),
        ('Recall', 'Recall'),
        ('Val Box Loss', 'Val Box Loss'),
        ('Train Box Loss', 'Train Box Loss'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Run Comparison', fontsize=16, fontweight='bold')

    colors = plt.cm.Set1(range(len(runs)))

    for ax, (metric_key, metric_title) in zip(axes.flatten(), metrics_to_plot):
        for i, run in enumerate(runs):
            epochs_data = run['data']['epochs']
            epoch_nums = [e['epoch'] for e in epochs_data]
            values = [e.get(metric_key, None) for e in epochs_data]

            # Filter out None values
            valid = [(ep, val) for ep, val in zip(epoch_nums, values) if val is not None]
            if valid:
                eps, vals = zip(*valid)
                ax.plot(eps, vals, label=run['name'][:25], color=colors[i % len(colors)],
                        linewidth=1.5, alpha=0.8)

        ax.set_title(metric_title, fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_title)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    plt.close()


def save_comparison_json(runs: List[Dict], output_path: str = "logs/run_comparison.json"):
    """Save comparison data to a JSON file for programmatic access."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    summary = []
    for run in runs:
        best = run['data']['best']
        entry = {
            'name': run['name'],
            'path': run['path'],
            'date': run['modified'],
            'total_epochs': run['data']['total_epochs'],
            'best_epoch': best.get('epoch', None),
            'best_mAP50': round(best.get('mAP50', 0), 4),
            'best_mAP50_95': round(best.get('mAP50-95', 0), 4),
            'best_precision': round(best.get('Precision', 0), 4),
            'best_recall': round(best.get('Recall', 0), 4),
        }
        if 'args' in run:
            entry['config'] = {k: run['args'].get(k) for k in
                               ['imgsz', 'batch', 'epochs', 'lr0', 'optimizer', 'patience']
                               if k in run['args']}
        summary.append(entry)

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Comparison JSON saved to: {output_path}")


# ============================================================
# CLI
# ============================================================

def main():
    """CLI entry point for comparing training runs."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare metrics across YOLO training runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all runs in default directory
  python compare_runs.py

  # Compare runs in a custom directory
  python compare_runs.py --runs-dir models/

  # Show detailed info for each run
  python compare_runs.py --detailed

  # Save comparison plot and JSON
  python compare_runs.py --output logs/comparison
        """
    )

    parser.add_argument('--runs-dir', nargs='+',
                        default=['runs/detect', 'models'],
                        help='Directories to scan for training runs (default: runs/detect models)')
    parser.add_argument('--output', default='logs/run_comparison',
                        help='Output path prefix for plots and JSON (default: logs/run_comparison)')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed per-run information')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--no-json', action='store_true',
                        help='Skip saving JSON summary')

    args = parser.parse_args()

    # Discover runs across all specified directories
    all_runs = []
    for run_dir in args.runs_dir:
        runs = discover_runs(run_dir)
        all_runs.extend(runs)

    if not all_runs:
        print("\nNo training runs found with results.csv files.")
        print(f"Searched in: {', '.join(args.runs_dir)}")
        print("\nMake sure you have completed at least one training run.")
        print("Training runs are stored by Ultralytics in runs/detect/ by default.")
        sys.exit(0)

    # De-duplicate by path
    seen_paths = set()
    unique_runs = []
    for run in all_runs:
        if run['path'] not in seen_paths:
            seen_paths.add(run['path'])
            unique_runs.append(run)

    # Print comparison table
    print_comparison_table(unique_runs)

    # Detailed view
    if args.detailed:
        print(f"\n{'='*60}")
        print("DETAILED RUN INFORMATION")
        print(f"{'='*60}")
        for run in unique_runs:
            print_detailed_run(run)

    # Plot
    if not args.no_plot:
        plot_comparison(unique_runs, f"{args.output}.png")

    # JSON
    if not args.no_json:
        save_comparison_json(unique_runs, f"{args.output}.json")


if __name__ == "__main__":
    main()

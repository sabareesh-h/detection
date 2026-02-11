"""
Model Export Utility
Export trained YOLOv11 model to various formats for deployment
"""

import os
from pathlib import Path

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


def export_model(
    model_path: str,
    format: str = 'onnx',
    imgsz: int = 640,
    half: bool = False,
    simplify: bool = True,
    output_dir: str = None
) -> str:
    """
    Export model to specified format
    
    Args:
        model_path: Path to trained .pt model
        format: Export format ('onnx', 'engine', 'torchscript', etc.)
        imgsz: Image size for export
        half: Use FP16 precision (for TensorRT)
        simplify: Simplify ONNX graph
        output_dir: Output directory (default: same as model)
        
    Returns:
        Path to exported model
    """
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("ultralytics not installed")
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Exporting to {format.upper()}...")
    
    export_args = {
        'format': format,
        'imgsz': imgsz,
    }
    
    if format == 'onnx':
        export_args['simplify'] = simplify
        export_args['dynamic'] = False
        export_args['opset'] = 12
        
    elif format == 'engine':  # TensorRT
        export_args['half'] = half
        export_args['workspace'] = 4  # GB
        
    elif format == 'torchscript':
        pass  # No special args
    
    result = model.export(**export_args)
    
    print(f"\nExport complete!")
    print(f"Exported model: {result}")
    
    return result


def benchmark_models(model_paths: list, imgsz: int = 640):
    """
    Benchmark inference speed of different model formats
    
    Args:
        model_paths: List of model file paths to benchmark
        imgsz: Image size for inference
    """
    import time
    import numpy as np
    
    print("\n" + "="*60)
    print("MODEL BENCHMARK")
    print("="*60)
    
    results = []
    dummy_image = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"Skipping (not found): {model_path}")
            continue
        
        model = YOLO(model_path)
        
        # Warmup
        for _ in range(5):
            model(dummy_image, verbose=False)
        
        # Benchmark
        times = []
        for _ in range(50):
            start = time.time()
            model(dummy_image, verbose=False)
            times.append((time.time() - start) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000 / avg_time
        
        # Get file size
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        results.append({
            'path': model_path,
            'format': Path(model_path).suffix,
            'size_mb': size_mb,
            'avg_ms': avg_time,
            'std_ms': std_time,
            'fps': fps
        })
        
        print(f"\n{Path(model_path).name}:")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  FPS:  {fps:.1f}")
    
    print("\n" + "="*60)
    return results


def main():
    """Export model from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export YOLOv11 model')
    parser.add_argument('model', help='Path to .pt model file')
    parser.add_argument('--format', '-f', default='onnx',
                       choices=['onnx', 'engine', 'torchscript', 'openvino'],
                       help='Export format')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--half', action='store_true',
                       help='FP16 precision (TensorRT)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark after export')
    
    args = parser.parse_args()
    
    exported_path = export_model(
        model_path=args.model,
        format=args.format,
        imgsz=args.imgsz,
        half=args.half
    )
    
    if args.benchmark:
        benchmark_models([args.model, exported_path], imgsz=args.imgsz)


if __name__ == "__main__":
    main()

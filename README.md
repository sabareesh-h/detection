# Defect Detection System

AI-powered defect detection using YOLOv11m with Basler camera integration.

## Project Structure

```
Image_detection/
├── config/
│   ├── dataset.yaml         # YOLO dataset configuration
│   └── system_config.json   # Camera and model settings
│
├── dataset/
│   ├── raw/                  # Raw captured images
│   ├── images/               # Split dataset for training
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/               # YOLO annotation files
│
├── scripts/
│   ├── camera_capture.py     # Basler camera utility
│   ├── validate_images.py    # Image quality validation
│   ├── split_dataset.py      # Dataset splitting
│   ├── train_model.py        # YOLOv11m training
│   ├── defect_detector.py    # Production inference
│   └── export_model.py       # Model export utility
│
├── models/                   # Trained model weights
├── logs/                     # Inspection logs
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: For Basler camera, install Pylon SDK from [Basler website](https://www.baslerweb.com/en/downloads/software-downloads/)

### 2. Collect Dataset

```bash
# Interactive capture mode
python scripts/camera_capture.py

# Keys: g=good, s=scratch, c=crack, d=dent, x=other, q=quit
```

### 3. Validate & Split Dataset

```bash
# Validate image quality
python scripts/validate_images.py dataset/raw

# Split into train/val/test
python scripts/split_dataset.py dataset/raw --output dataset
```

### 4. Annotate Images

Use [Roboflow](https://roboflow.com) or [LabelImg](https://github.com/HumanSignal/labelImg):
1. Upload images
2. Draw bounding boxes around defects
3. Export in YOLO format

### 5. Train Model

```bash
python scripts/train_model.py --data config/dataset.yaml --epochs 100
```

### 6. Run Inference

```bash
# Single image
python scripts/defect_detector.py --image path/to/image.png --model models/best.pt

# Continuous camera mode
python scripts/defect_detector.py --model models/best.pt --continuous
```

### 7. Export for Production

```bash
# Export to ONNX
python scripts/export_model.py models/best.pt --format onnx
```

## Configuration

### config/dataset.yaml
```yaml
path: C:/path/to/dataset
train: images/train
val: images/val
names:
  0: scratch
  1: crack
  2: dent
nc: 3
```

### config/system_config.json
```json
{
  "camera": {
    "exposure_time_us": 15000,
    "gain_db": 0
  },
  "model": {
    "confidence_threshold": 0.5
  }
}
```

## Workflow Summary

1. **Capture** → Basler camera captures product images
2. **Annotate** → Label defects with bounding boxes
3. **Train** → YOLOv11m learns to detect defects
4. **Deploy** → Real-time inspection in production

## API Reference

```python
from scripts.defect_detector import DefectDetector

detector = DefectDetector('models/best.pt', conf_threshold=0.5)
result = detector.detect_from_file('product.png')

if result['is_defective']:
    print(f"REJECT: {result['defects']}")
else:
    print("PASS")
```

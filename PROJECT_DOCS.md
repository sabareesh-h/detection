# 📁 Defect Detection System — Project Documentation

> **AI Agent Instructions**: This file is maintained by an AI agent. When changes occur in the project, update the relevant sections below. Follow the structure strictly. Keep the `Last Updated` timestamps current.

---

## 📌 Project Overview

- **Project Name**: `Defect Detection System`
- **Description**: AI-powered visual defect detection using YOLO (Ultralytics) with Basler camera integration for real-time production-line inspection.
- **Model**: `YOLOv11m` (medium variant) — also tested with `YOLO26n` (nano)
- **Version**: `0.2.0`
- **Status**: `In Progress`
- **Last Updated**: `2026-03-04`
- **Repository**: [sabareesh-h/detection](https://github.com/sabareesh-h/detection)

---

## 🎯 Detection Classes

| Class ID | Class Name   | Description                                |
|----------|--------------|--------------------------------------------|
| 0        | `Good`       | No defect detected — product passes QC     |
| 1        | `Flat_line`  | Flat-line defect on the product surface     |
| 2        | `Unwash`     | Unwashed / residue defect on the product    |

> **Agent Note**: Update this table if new defect classes are added in `config/dataset.yaml` and `config/system_config.json`.

---

## 🛠️ Tech Stack

| Layer              | Technology          | Version     |
|--------------------|---------------------|-------------|
| Language           | Python              | 3.14+       |
| ML Framework       | Ultralytics (YOLO)  | ≥ 8.3.0    |
| Deep Learning      | PyTorch             | ≥ 2.0.0    |
| Vision Library     | torchvision         | ≥ 0.15.0   |
| Camera SDK         | pypylon (Basler)    | ≥ 3.0.0    |
| Image Processing   | OpenCV              | ≥ 4.8.0    |
| Image I/O          | Pillow              | ≥ 10.0.0   |
| Data Processing    | NumPy / Pandas      | ≥ 1.24 / ≥ 2.0 |
| Dataset Splitting  | scikit-learn        | ≥ 1.3.0    |
| Visualization      | matplotlib          | ≥ 3.7.0    |
| Dataset Management | Roboflow            | ≥ 1.1.0    |
| Annotation Tool    | CVAT (Docker-based) | —           |
| Containerization   | Docker / Docker Compose | — |

> **Agent Note**: Update this table whenever a dependency is added, removed, or upgraded in `requirements.txt`.

---

## 📂 Project Structure

```
Image_detection/
├── config/
│   ├── dataset.yaml              # YOLO dataset config (paths, classes)
│   ├── system_config.json        # Camera, model, and inspection settings
│   └── hyperparams.yaml          # Hyperparameter tuning presets
│
├── dataset/
│   ├── raw/                      # Raw captured images (unprocessed)
│   ├── images/
│   │   ├── train/                # Training images
│   │   ├── val/                  # Validation images
│   │   └── test/                 # Test images
│   └── labels/
│       ├── train/                # YOLO-format annotation TXT files (train)
│       ├── val/                  # YOLO-format annotation TXT files (val)
│       └── test/                 # YOLO-format annotation TXT files (test)
│
├── scripts/
│   ├── camera_capture.py         # Basler camera capture utility
│   ├── validate_images.py        # Image quality validation
│   ├── split_dataset.py          # Train/val/test splitting
│   ├── download_dataset.py       # Download dataset from Roboflow
│   ├── prepare_dataset.py        # Dataset preparation pipeline
│   ├── augment_dataset.py        # Offline dataset augmentation (albumentations)
│   ├── train_model.py            # YOLOv11m training script
│   ├── evaluate_model.py         # Full model evaluation report
│   ├── compare_runs.py           # Training run metrics comparison
│   ├── defect_detector.py        # Production inference pipeline
│   ├── export_model.py           # Model export (ONNX, TensorRT, etc.)
│   └── webcam_demo.py            # Quick webcam inference demo
│
├── runs/                         # Ultralytics training/val/detect outputs
│   └── detect/
│       ├── models/               # Saved model weights
│       ├── runs/                 # Training run logs & metrics
│       └── val/                  # Validation results
│
├── cvat/                         # CVAT annotation tool (Docker setup)
├── yolo11m.pt                    # Pre-trained YOLO11m weights
├── yolo26n.pt                    # Pre-trained YOLO26n weights
├── requirements.txt              # Python dependencies
├── start_cvat.bat                # Start CVAT annotation server
├── stop_cvat.bat                 # Stop CVAT annotation server
├── model_training_workflow_guide.md  # Detailed training guide
├── Understanding_YOLO_Complete_Guide.docx  # YOLO concepts reference
├── PROJECT_DOCS.md               # This file
├── PROBLEMS_AND_SOLUTIONS.md     # Problems faced & solutions log
└── README.md                     # Quick-start entry point
```

> **Agent Note**: Reflect any new folders or files added to the project here.

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/sabareesh-h/detection.git
cd Image_detection

# 2. Create and activate virtual environment
python -m venv defect_env
# Windows:
defect_env\Scripts\activate
# Linux/macOS:
source defect_env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) GPU environment — requires CUDA-compatible PyTorch
python -m venv defect_env_gpu
defect_env_gpu\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Install Basler Pylon SDK (for camera integration)
# Download from: https://www.baslerweb.com/en/downloads/software-downloads/
```

> **Agent Note**: Two virtual environments exist — `defect_env` (CPU) and `defect_env_gpu` (GPU/CUDA). Update commands if setup process changes.

---

## 🔧 Configuration

### `config/dataset.yaml` — Dataset Configuration

```yaml
path: C:/Users/RohithSuryaCKM/Downloads/Projects/Image_detection/dataset
train: images/train
val: images/val
test: images/test

names:
  0: Good
  1: Flat_line
  2: Unwash

nc: 3
```

### `config/system_config.json` — System Configuration

| Section       | Parameter              | Value        | Description                          |
|---------------|------------------------|--------------|--------------------------------------|
| **Camera**    | `exposure_time_us`     | `15000`      | Camera exposure time (microseconds)  |
|               | `gain_db`              | `0`          | Camera gain (dB)                     |
|               | `pixel_format`         | `Mono8`      | Grayscale 8-bit pixel format         |
|               | `trigger_mode`         | `Software`   | Software-triggered capture           |
|               | `timeout_ms`           | `5000`       | Capture timeout (milliseconds)       |
| **Model**     | `weights_path`         | `models/best.pt` | Trained model weights path       |
|               | `confidence_threshold` | `0.5`        | Minimum detection confidence         |
|               | `iou_threshold`        | `0.45`       | IoU threshold for NMS                |
|               | `image_size`           | `640`        | Input image size for inference       |
| **Inspection**| `save_images`          | `true`       | Save inspection images               |
|               | `save_path`            | `logs/inspections` | Where to save images            |
|               | `log_to_database`      | `true`       | Enable SQLite logging                |
|               | `database_path`        | `logs/inspections.db` | SQLite database path          |

> **Agent Note**: Update this table when config values change.

---

## 🔄 Workflow Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  1. CAPTURE  │───▶│ 2. ANNOTATE  │───▶│  3. TRAIN    │───▶│  4. DEPLOY   │
│  Basler Cam  │    │  CVAT / RF   │    │  YOLOv11m    │    │  Real-time   │
│  + Validate  │    │  YOLO format │    │  GPU/CPU     │    │  Inspection  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Step-by-Step

| Step | Script / Tool              | Command                                              |
|------|----------------------------|------------------------------------------------------|
| 1    | Capture images             | `python scripts/camera_capture.py`                   |
| 2    | Validate image quality     | `python scripts/validate_images.py dataset/raw`      |
| 3    | Download dataset (Roboflow)| `python scripts/download_dataset.py`                 |
| 4    | Prepare dataset            | `python scripts/prepare_dataset.py`                  |
| 5    | Split train/val/test       | `python scripts/split_dataset.py dataset/raw --output dataset` |
| 6    | Augment training data      | `python scripts/augment_dataset.py --input dataset/images/train --labels dataset/labels/train --multiplier 5` |
| 7    | Annotate (CVAT)            | `start_cvat.bat` → open `http://localhost:8080`      |
| 8    | Train model                | `python scripts/train_model.py --data config/dataset.yaml --epochs 100` |
| 9    | Evaluate model             | `python scripts/evaluate_model.py --model models/best.pt` |
| 10   | Compare training runs      | `python scripts/compare_runs.py` |
| 11   | Run inference (single)     | `python scripts/defect_detector.py --image path/to/img.png --model models/best.pt` |
| 12   | Run inference (continuous) | `python scripts/defect_detector.py --model models/best.pt --continuous` |
| 13   | Export model               | `python scripts/export_model.py models/best.pt --format onnx` |
| 14   | Webcam demo                | `python scripts/webcam_demo.py`                      |

> **Agent Note**: Add new pipeline steps or scripts as they are created.

---

## 📜 Script Reference

### `camera_capture.py`
- Interfaces with **Basler industrial cameras** via pypylon SDK
- Provides `BaslerCamera` class and `MockCamera` fallback for testing
- Interactive capture mode: `g`=good, `s`=scratch, `c`=crack, `d`=dent, `x`=other, `q`=quit

### `validate_images.py`
- Validates image quality (corruption, resolution, file integrity)
- Reports statistics on the dataset before training

### `split_dataset.py`
- Splits raw dataset into **train / val / test** sets
- Uses scikit-learn's stratified splitting to maintain class balance

### `download_dataset.py`
- Downloads datasets from **Roboflow** using the Roboflow API

### `prepare_dataset.py`
- Full dataset preparation pipeline — combines download, validate, and organize

### `augment_dataset.py`
- Offline augmentation using **albumentations** library
- 3 intensity presets: `light`, `medium`, `heavy`
- Preserves YOLO bounding box annotations through augmentations
- Augmentations: rotation, brightness/contrast, Gaussian noise/blur, CLAHE, random shadow, HSV jitter
- **Preview mode** (`--preview`) to visualize augmentations before saving
- Generates N augmented copies per image (`--multiplier`)

### `train_model.py`
- Trains **YOLOv11m** on the defect detection dataset
- Key parameters: `epochs=100`, `batch_size=16`, `img_size=640`, `patience=20`
- Supports **GPU training** (auto-detects CUDA) and **resume from checkpoint**
- Includes `validate_model()` function for post-training evaluation

### `evaluate_model.py`
- Comprehensive model evaluation report generator
- Runs Ultralytics validation: **mAP@50, mAP@50-95, Precision, Recall, F1** (overall + per-class)
- Generates **confusion matrix** and **confidence histogram**
- Benchmarks **inference speed** (FPS, latency breakdown) on current hardware
- Saves full report as **JSON + Markdown** to `logs/evaluation_report_<timestamp>/`

### `compare_runs.py`
- Scans `runs/detect/` and `models/` for Ultralytics training runs
- Parses `results.csv` and extracts per-epoch metrics
- Prints **comparison table** (mAP, precision, recall per run)
- Generates **overlay plots** (loss, mAP curves) and saves **JSON summary**

### `defect_detector.py`
- Production inference pipeline with 3 main components:
  - **`DefectDetector`** — Core detection class (loads model, runs inference, draws results)
  - **`InspectionLogger`** — Logs results to SQLite database
  - **`ProductionInspectionSystem`** — End-to-end system combining camera + detection + logging

### `export_model.py`
- Exports trained model to production formats: **ONNX**, **TensorRT**, **CoreML**, etc.

### `webcam_demo.py`
- Quick demo script — runs detection on webcam feed

---

## 🧠 Model Details

| Property              | Value                             |
|-----------------------|-----------------------------------|
| Architecture          | YOLOv11m (medium)                 |
| Input Size            | 640 × 640                         |
| Pre-trained Weights   | `yolo11m.pt` (COCO pre-trained)   |
| Alt. Weights          | `yolo26n.pt` (nano, for speed)    |
| Training Epochs       | 100 (default)                     |
| Batch Size            | 16                                |
| Early Stopping        | patience = 20 epochs              |
| Optimizer             | Ultralytics default (SGD/AdamW)   |
| Confidence Threshold  | 0.5                               |
| IoU Threshold (NMS)   | 0.45                              |
| Output                | Bounding boxes + class + confidence |

### Training Augmentations (via Ultralytics Engine)
- Mosaic augmentation
- MixUp augmentation
- CutMix augmentation
- Random flips, scaling, color jitter
- Auto-augmentation policies

> **Agent Note**: Update model details when retraining with different hyperparameters.

---

## 🚀 Features

- [x] Basler camera integration for image capture
- [x] Image quality validation pipeline
- [x] Dataset splitting (stratified train/val/test)
- [x] Roboflow dataset download integration
- [x] CVAT annotation tool setup (Docker)
- [x] YOLOv11m training with GPU support
- [x] Production inference pipeline (`DefectDetector`)
- [x] SQLite inspection logging (`InspectionLogger`)
- [x] Full production system with camera integration (`ProductionInspectionSystem`)
- [x] Model export (ONNX, TensorRT, etc.)
- [x] Webcam demo mode
- [ ] Dashboard / UI for monitoring inspection results
- [ ] Multi-camera support
- [ ] Model versioning and A/B testing
- [ ] Automated retraining pipeline

> **Agent Note**: Check off features as they are completed. Add new features as planned.

---

## 🐛 Known Issues

| Issue | Severity | Status |
|-------|----------|--------|
| Training stopped early due to small dataset size — low patience value | Medium | Resolved (patience adjusted) |
| GPU environment requires separate venv (`defect_env_gpu`) with CUDA PyTorch | Low | Documented |
| Basler SDK (`pypylon`) requires manual download of Pylon SDK | Low | Documented |

> **Agent Note**: Add new bugs here as discovered. Update status as they are resolved.

---

## 📋 Changelog

### [Unreleased]
- Exploring additional defect classes
- Dashboard for inspection monitoring

### v0.2.0 — `2026-02-27`
- Added Roboflow dataset download script
- Added dataset preparation pipeline
- GPU training environment setup
- Augmentation engine exploration (Mosaic, MixUp, CutMix)
- Updated dataset classes: `Good`, `Flat_line`, `Unwash`

### v0.1.0 — `2026-02-11`
- Initial project setup
- Camera capture, validation, and splitting scripts
- YOLOv11m training script
- Production inference pipeline
- CVAT annotation tool integration
- Model export utility

> **Agent Note**: Add entries under `[Unreleased]` for every meaningful change. Move to versioned section on each release.

---

## 👥 Contributors

| Name | Role | Contact |
|------|------|---------|
| Sabareesh H | Lead Developer | [GitHub](https://github.com/sabareesh-h) |

---

## 📄 License

See `LICENSE` file for details.

---

*This document is automatically maintained. Manual edits should follow the structure above.*

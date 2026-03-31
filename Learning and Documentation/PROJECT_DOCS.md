# 📁 Defect Detection System — Project Documentation

> **AI Agent Instructions**: This file is maintained by an AI agent. When changes occur in the project, update the relevant sections below. Follow the structure strictly. Keep the `Last Updated` timestamps current.

---

## 📌 Project Overview

- **Project Name**: `Defect Detection System`
- **Description**: AI-powered visual defect detection using YOLO (Ultralytics) with Basler camera integration for real-time production-line inspection.
- **Model**: `YOLOv11m` (medium variant) — also tested with `YOLO26n` (nano)
- **Version**: `0.2.0`
- **Status**: `In Progress`
- **Last Updated**: `2026-03-24`
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
| Language           | Python              | 3.11 (GPU env) |
| ML Framework       | Ultralytics (YOLO)  | 8.4.19      |
| Deep Learning      | PyTorch             | 2.10.0+cu128 |
| Vision Library     | torchvision         | 0.25.0+cu128 |
| CUDA Toolkit       | NVIDIA CUDA         | 12.8 (cu128) |
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
├── dataset/                      # ⚠️ UNUSED — kept as archive only
│                                 # All active data is in scripts/dataset/
│
├── scripts/
│   ├── camera_capture.py         # Basler camera capture utility
│   ├── validate_images.py        # Image quality validation
│   ├── split_dataset.py          # Train/val/test splitting
│   ├── download_dataset.py       # Download dataset from Roboflow
│   ├── prepare_dataset.py        # Dataset preparation + split pipeline
│   ├── augment_dataset.py        # Offline dataset augmentation (albumentations)
│   ├── train_model.py            # YOLOv11m training script
│   ├── evaluate_model.py         # Full model evaluation report
│   ├── compare_runs.py           # Training run metrics comparison
│   ├── run_pipeline.py           # ⭐ Master pipeline (chains all steps)
│   ├── defect_detector.py        # Production inference pipeline
│   ├── export_model.py           # Model export (ONNX, TensorRT, etc.)
│   ├── webcam_demo.py            # Quick webcam inference demo
│   └── dataset/                  # ⭐ SINGLE dataset location (all active data)
│       ├── raw/                  # camera_capture.py saves here
│       ├── images/
│       │   ├── train/            # Training images (originals + augmented)
│       │   ├── val/              # Validation images (originals only)
│       │   └── test/             # Test images (originals only)
│       └── labels/
│           ├── train/            # YOLO .txt labels (train)
│           ├── val/              # YOLO .txt labels (val)
│           └── test/             # YOLO .txt labels (test)
│
├── Learning and Documentation/
│   ├── PROJECT_DOCS.md           # Full project documentation (this file)
│   ├── PROBLEMS_AND_SOLUTIONS.md # Problems faced & solutions log
│   ├── file_flow.md              # ⭐ How files move from capture → detection
│   ├── Coding_understanding.md   # Line-by-line code explanations
│   ├── camera_technical_specifications_guide.md
│   ├── comprehensive_project_roadmap.md
│   ├── model_training_workflow_guide.md
│   ├── Code for yolo training.md
│   ├── Difference YOLO and Keyence.md
│   ├── Understanding_YOLO_Complete_Guide.docx
│   ├── yolo_beginner_guide.md
│   ├── yolo_complete_guide.md
│   └── understanding_code/       # Script-by-script explanations
│       ├── Camera_capture.md
│       └── Validate_images.md
│
├── runs/                         # Ultralytics training/val/detect outputs
│   └── detect/
│       ├── models/               # Saved model weights
│       ├── runs/                 # Training run logs & metrics
│       └── val/                  # Validation results
│
├── cvat/                         # CVAT annotation tool (Docker setup)
├── requirements.txt              # Python dependencies
├── start_cvat.bat                # Start CVAT annotation server
├── stop_cvat.bat                 # Stop CVAT annotation server
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

# 4. GPU environment — requires Python 3.11 + CUDA 12.8 PyTorch
#    ⚠️ Blackwell GPUs (RTX PRO 500, RTX 50-series) REQUIRE cu128!
"C:\Users\RohithSuryaCKM\AppData\Local\Programs\Python\Python311\python.exe" -m venv defect_env_gpu311
defect_env_gpu311\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics opencv-python numpy pyyaml matplotlib pandas seaborn polars scipy

# 5. Install Basler Pylon SDK (for camera integration)
# Download from: https://www.baslerweb.com/en/downloads/software-downloads/
```

> **Agent Note**: GPU environment is `defect_env_gpu311` (Python 3.11 + CUDA 12.8). Blackwell GPUs require `cu128` — lower CUDA versions (cu121/cu124/cu126) do NOT support sm_120.

---

## 🔧 Configuration

### `config/dataset.yaml` — Dataset Configuration

```yaml
path: C:/Users/RohithSuryaCKM/Downloads/Projects/Image_detection/scripts/dataset
train: images/train
val: images/val
test: images/test

names:
  0: Good(Top)
  1: Rust(Top)
  2: Rust(Mid)
  3: Rust(Bottom)
  4: Rust(Thread)
  5: Good(Mid)
  6: Good(Thread)
  7: Good(Bottom)

nc: 8
```

> **Note**: `path` points to `scripts/dataset/` (updated 2026-03-24). The root `dataset/` folder is an unused archive.

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

## 🔄 Workflow Pipeline — Full Integration View

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DEFECT DETECTION PIPELINE                          │
│                                                                        │
│  PHASE 1: DATA COLLECTION                                             │
│  ┌──────────────────┐     ┌──────────────────┐                        │
│  │ camera_capture.py │────▶│validate_images.py│                        │
│  │ (Basler camera)   │     │ (quality check)  │                        │
│  └──────────────────┘     └────────┬─────────┘                        │
│                                    │                                   │
│  PHASE 2: DATA PREPARATION                                            │
│  ┌──────────────────┐     ┌────────▼─────────┐    ┌────────────────┐  │
│  │download_dataset.py│────▶│prepare_dataset.py│───▶│ split_dataset  │  │
│  │ (Roboflow data)   │     │ (merge + clean)  │    │  (70/15/15)    │  │
│  └──────────────────┘     └──────────────────┘    └───────┬────────┘  │
│                                                           │            │
│  PHASE 3: AUGMENTATION                                    │            │
│  ┌──────────────────┐                                     │            │
│  │augment_dataset.py │◄───────────────────────────────────┘            │
│  │ (5x multiplier)   │   87 images ──▶ ~435 images                    │
│  └────────┬─────────┘                                                  │
│           │                                                            │
│  PHASE 4: TRAINING                                                     │
│  ┌────────▼─────────┐     ┌──────────────────┐                        │
│  │  train_model.py   │     │ hyperparams.yaml │  ◀── pick a preset    │
│  │  (YOLOv11m)       │◄────│ small_dataset    │                        │
│  └────────┬─────────┘     └──────────────────┘                        │
│           │                                                            │
│  PHASE 5: EVALUATION                                                   │
│  ┌────────▼─────────┐     ┌──────────────────┐                        │
│  │evaluate_model.py  │     │ compare_runs.py  │                        │
│  │ (mAP, F1, speed)  │     │ (compare exps)   │                        │
│  └────────┬─────────┘     └──────────────────┘                        │
│           │                                                            │
│  PHASE 6: DEPLOYMENT                                                   │
│  ┌────────▼─────────┐     ┌──────────────────┐                        │
│  │defect_detector.py │     │ export_model.py  │                        │
│  │ (production run)  │     │ (ONNX/TensorRT)  │                        │
│  └──────────────────┘     └──────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### ⭐ Master Pipeline — Run Everything with One Command

`run_pipeline.py` chains all the above steps automatically.

| Mode | What it Does | Command |
|------|-------------|----------|
| **full** | Validate → Augment → Train → Evaluate → Compare → Export | `python scripts/run_pipeline.py --mode full --preset small_dataset` |
| **train-eval** | Train → Evaluate → Compare (skip augmentation) | `python scripts/run_pipeline.py --mode train-eval --preset baseline` |
| **eval-only** | Evaluate an existing model | `python scripts/run_pipeline.py --mode eval-only --model models/best.pt` |
| **augment-only** | Augment training data only | `python scripts/run_pipeline.py --mode augment-only --multiplier 10` |

**Recommended first run** (for your small dataset):
```bash
python scripts/run_pipeline.py --mode full --preset small_dataset --multiplier 5
```

**Quick sanity check** (fast, 20 epochs — just to verify the pipeline works):
```bash
python scripts/run_pipeline.py --mode full --preset fast_training --multiplier 2
```

### Individual Step Commands

| Step | Script / Tool              | Command                                              |
|------|----------------------------|------------------------------------------------------|
| 1    | Capture images             | `python scripts/camera_capture.py`                   |
| 2    | Annotate in CVAT           | `start_cvat.bat` → open `http://localhost:8080`      |
| 3    | Prepare + split dataset    | `python scripts/prepare_dataset.py`                  |
| 4    | Augment **train only**     | `python scripts/augment_dataset.py --input scripts/dataset/images/train --labels scripts/dataset/labels/train --multiplier 3` |
| 5    | Validate image quality     | `python scripts/validate_images.py scripts/dataset/raw` |
| 7    | Annotate (CVAT)            | `start_cvat.bat` → open `http://localhost:8080`      |
| 8    | Train model                | `python scripts/train_model.py --data config/dataset.yaml --epochs 100` |
| 9    | Evaluate model             | `python scripts/evaluate_model.py --model models/best.pt` |
| 10   | Compare training runs      | `python scripts/compare_runs.py` |
| 11   | Run inference (single)     | `python scripts/defect_detector.py --image path/to/img.png --model models/best.pt` |
| 12   | Run inference (continuous) | `python scripts/defect_detector.py --model models/best.pt --continuous` |
| 13   | Export model               | `python scripts/export_model.py models/best.pt --format onnx` |
| 14   | Webcam demo                | `python scripts/webcam_demo.py`                      |

### 🔁 Iteration Cycle

After each training run, follow this cycle to continuously improve:

```
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │   Augment ──▶ Train ──▶ Evaluate ──▶ Analyze Results ──┐    │
  │                                                         │    │
  │   ◀── Adjust hyperparams / Add more data ◀──────────────┘    │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
```

**What to check after each run:**
1. **Confusion matrix** → Are `Flat_line` and `Unwash` being confused with each other?
2. **Confidence histogram** → Is the 0.5 threshold too high/low for your data?
3. **Per-class F1** → Which class is weakest? Collect more data for that class.
4. **Training loss curve** → Still decreasing? Train longer. Plateaued? Change hyperparams.

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
- Preserves YOLO bounding box annotations through augmentations
- **Current augmentations** (updated 2026-03-24):
  - `Rotate(±15°, p=0.5)` — simulates part placement angle variation
  - `HorizontalFlip(p=0.5)` — parts can appear mirrored
  - `VerticalFlip(p=0.3)` — parts can be placed inverted
  - `GaussNoise(p=0.4)` — simulates Basler camera sensor noise
  - `GaussianBlur(p=0.3)` — simulates slight camera focus variation
- **CLAHE removed**: images are already greyscaled with rust-aware CLAHE during capture; re-applying would bias the model toward over-enhanced contrast
- **Preview mode** (`--preview`) to visualize augmentations before saving
- Generates N augmented copies per image (`--multiplier`, default=3)
- ⚠️ **Run AFTER `prepare_dataset.py`** — augment only `images/train/`, never val or test

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

### `run_pipeline.py`
- ⭐ **Master pipeline** — chains all steps into a single command
- 4 modes: `full`, `train-eval`, `eval-only`, `augment-only`
- Loads hyperparameter presets from `config/hyperparams.yaml`
- Prints final summary with mAP, F1, elapsed time
- Recommended: `python scripts/run_pipeline.py --mode full --preset small_dataset`

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
| Batch Size            | 8 (GPU: RTX PRO 500, 6GB VRAM)    |
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
| GPU environment requires `defect_env_gpu311` (Python 3.11) with CUDA 12.8 PyTorch | Low | Documented |
| Blackwell GPU (RTX PRO 500) needs `cu128` — cu121/cu124/cu126 fail with "no kernel image" | High | Resolved |
| Basler SDK (`pypylon`) requires manual download of Pylon SDK | Low | Documented |
| Two `dataset/` folders caused confusion — root `dataset/` and `scripts/dataset/` | Medium | Resolved (config unified to `scripts/dataset/`) |
| CLAHE in augment_dataset.py was redundant after greyscale capture — over-enhanced contrast | Medium | Resolved (CLAHE removed, replaced with flips + noise + blur) |

> **Agent Note**: Add new bugs here as discovered. Update status as they are resolved.

---

## 📋 Changelog

### [Unreleased]
- Dashboard for inspection monitoring

### v0.3.0 — `2026-03-24`
- **Unified dataset path**: `config/dataset.yaml` and `prepare_dataset.py` both now point to `scripts/dataset/` — eliminates dual-folder confusion
- **Updated augmentation pipeline**: removed CLAHE (redundant post-greyscale), added `HorizontalFlip`, `VerticalFlip`, `GaussNoise`, `GaussianBlur`
- **Greyscale integration**: `camera_capture.py` now applies rust-aware greyscale + CLAHE at capture time (choose original or greyscale mode at startup)
- **Documented correct workflow order**: `prepare_dataset.py` first (split), then `augment_dataset.py` on train only — val/test remain original
- Added `Learning and Documentation/file_flow.md` — visual map of how files move from capture to detection

### v0.2.1 — `2026-03-05`
- Fixed Blackwell GPU support — requires PyTorch cu128 (CUDA 12.8)
- Updated GPU environment to `defect_env_gpu311` (Python 3.11)
- Verified GPU training: RTX PRO 500, batch=4, 20.4ms/image inference
- Recommended batch size: 8 (6 GB VRAM)

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


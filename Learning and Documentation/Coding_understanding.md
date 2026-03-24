# 📖 Coding Understanding — Line-by-Line Explanation

> This document provides a detailed, beginner-friendly explanation of every script in the project.  
> Each section covers: **what the script does**, **how it works**, and **key concepts** to help you understand the code.

---

## Table of Contents

1. [camera_capture.py](#1-camera_capturepy) — Camera hardware integration
2. [validate_images.py](#2-validate_imagespy) — Image quality checking
3. [download_dataset.py](#3-download_datasetpy) — Roboflow data download
4. [prepare_dataset.py](#4-prepare_datasetpy) — Merging and organizing data
5. [split_dataset.py](#5-split_datasetpy) — Train/val/test splitting
6. [augment_dataset.py](#6-augment_datasetpy) — Offline data augmentation
7. [train_model.py](#7-train_modelpy) — YOLO model training
8. [evaluate_model.py](#8-evaluate_modelpy) — Model evaluation & reporting
9. [compare_runs.py](#9-compare_runspy) — Training run comparison
10. [defect_detector.py](#10-defect_detectorpy) — Production inference
11. [export_model.py](#11-export_modelpy) — Model format export
12. [webcam_demo.py](#12-webcam_demopy) — Quick webcam demo
13. [run_pipeline.py](#13-run_pipelinepy) — Master pipeline
14. [Config Files](#14-config-files) — dataset.yaml, system_config.json, hyperparams.yaml

---

## 1. `camera_capture.py`

**Purpose**: Interfaces with **Basler industrial cameras** to capture product images for the dataset.

### Imports & Setup (Lines 1–19)

```python
from pypylon import pylon    # Basler camera SDK (hardware driver)
import cv2                    # OpenCV for image handling
import numpy as np            # NumPy for array operations
```

- `pypylon` is wrapped in a `try/except` — if the Basler SDK isn't installed, the script still loads but sets `PYPYLON_AVAILABLE = False`
- This pattern is called **graceful degradation** — the code works even without the hardware SDK

### `BaslerCamera` Class (Lines 22–184)

This is the main class that talks to the physical camera.

#### `__init__` — Initialization
```python
def __init__(self, config_path: str = None):
    self.config = self._load_config(config_path)   # Load camera settings (exposure, gain)
    self.camera = None                              # Camera object (set later on connect)
    self.converter = None                           # Converts raw camera format → OpenCV BGR
```
**Why?** The camera isn't connected on creation — you decide when to connect with `.connect()`. The config is loaded from `system_config.json`.

#### `_load_config` — Loading Camera Settings
```python
default_config = {
    "exposure_time_us": 15000,   # How long the sensor collects light (15ms)
    "gain_db": 0,                # Signal amplification (0 = no artificial boost)
    "pixel_format": "Mono8",     # Grayscale, 8-bit per pixel
    "trigger_mode": "Software",  # Capture triggered by code, not external signal
    "timeout_ms": 5000           # Wait max 5 seconds for a capture
}
```
**Key concept**: If no config file exists, these defaults are used. This prevents crashes in new setups.

#### `connect` — Connecting to Camera Hardware
```python
self.camera = pylon.InstantCamera(
    pylon.TlFactory.GetInstance().CreateFirstDevice()  # Find first connected Basler camera
)
self.camera.Open()                                     # Open connection to the camera
```
**What happens**: Pylon SDK scans USB/GigE for Basler cameras → grabs the first one → opens a connection → applies exposure/gain settings.

#### `capture` — Taking a Photo
```python
self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)   # Begin capturing
grab_result = self.camera.RetrieveResult(timeout_ms, ...)       # Wait for a frame
image = self.converter.Convert(grab_result)                      # Convert to BGR
img_array = image.GetArray()                                     # → NumPy array
```
**Flow**: Start grab → Wait for frame → Convert raw sensor data → Return as NumPy array (usable with OpenCV).

#### `capture_and_save` — Capture + Save to Disk
```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")   # Unique filename from timestamp
filename = f"{label}_{timestamp}.png"                       # e.g., "good_20260216_143021_123456.png"
cv2.imwrite(str(filepath), image)                           # Save as PNG (lossless)
```
**Why PNG?** Lossless compression preserves every pixel — important for training data. JPEG would add compression artifacts.

#### Context Manager (`__enter__` / `__exit__`)
```python
with camera:    # Automatically calls connect() on enter, disconnect() on exit
    image = camera.capture()
```
**Why?** Ensures the camera is always properly disconnected, even if an error occurs. It's a Python best practice.

### `MockCamera` Class (Lines 187–233)

**Purpose**: A fake camera for testing when no physical Basler camera is available.

```python
def capture(self) -> np.ndarray:
    image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)  # Random noise image
    if np.random.random() > 0.5:
        cv2.rectangle(image, (x, y), (x+50, y+20), (50, 50, 50), -1)   # Add fake "defect"
    return image
```
**Why?** You can develop and test the full pipeline on any computer, even without a Basler camera. The `get_camera()` factory function auto-selects `MockCamera` when pypylon isn't available.

### `collect_dataset_interactive` (Lines 252–295)

Interactive collection loop:
```python
key = input("Enter label key (g/s/c/d/x) or q to quit: ")
label_map = {'g': 'good', 's': 'scratch', 'c': 'crack', 'd': 'dent', 'x': 'other'}
camera.capture_and_save(output_dir, label)
```
**How it works**: User presses a key → script maps it to a class label → captures from camera → saves to `dataset/raw/{label}/`.

---

## 2. `validate_images.py`

**Purpose**: Checks image quality before training — filters out blurry, dark, corrupt, or overexposed images.

### `ImageQualityValidator` Class

#### Quality Checks Performed

```python
# Check 1: Resolution — is the image big enough?
if width < 640 or height < 480:
    issues.append("Low resolution")

# Check 2: Brightness — mean pixel value (0=black, 255=white)
mean_brightness = np.mean(gray)              # Average pixel intensity
# Acceptable range: 50–205 (not too dark, not too bright)

# Check 3: Contrast — standard deviation of pixel values
std_contrast = np.std(gray)                  # Higher = more variation = better contrast
# Below 20.0 means the image is too flat/uniform

# Check 4: Sharpness — Laplacian variance
laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
# Laplacian detects edges. High variance = sharp image. Low = blurry.
# Below 100.0 means the image is too blurry

# Check 5: Clipping — over/under exposure
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
underexposed = hist[0][0] / pixel_count      # % of pixels that are pure black
overexposed = hist[255][0] / pixel_count     # % of pixels that are pure white
# If >10% of pixels are clipped, the lighting is wrong
```

**Key concept — Laplacian for sharpness**:  
The Laplacian operator finds edges by calculating the second derivative of pixel intensities. A sharp image has many strong edges → high variance. A blurry image has weak edges → low variance.

### `validate_directory` — Batch Validation
```python
patterns = [f"{directory}/**/*.{ext}" for ext in ['png', 'jpg', 'jpeg']]
# Uses glob with ** to find images recursively in all subdirectories
```
Scans an entire folder, validates each image, counts pass/fail, and reports problem images.

---

## 3. `download_dataset.py`

**Purpose**: Downloads a pre-labeled dataset from **Roboflow** (a dataset hosting platform).

```python
rf = Roboflow(api_key=api_key)                              # Connect to Roboflow with your API key
project = rf.workspace(workspace).project(project)           # Navigate to the project
dataset = project.version(version).download("yolov11")       # Download in YOLOv11 format
```

**How Roboflow works**:
1. You upload images and annotate them on Roboflow's web UI
2. Roboflow hosts them and provides an API
3. This script downloads them in YOLO format (images + label `.txt` files)

**API key handling**: Accepts key via `--api-key` flag, `ROBOFLOW_API_KEY` environment variable, or interactive prompt. This is a **security best practice** — never hardcode API keys.

---

## 4. `prepare_dataset.py`

**Purpose**: Merges images and CVAT-exported YOLO annotations from multiple sources into a unified dataset structure.

### Configuration Block (Lines 13–47)
```python
ANNOTATION_SOURCES = [
    {
        "labels_dir": PROJECT_ROOT / "task_4_annotations.../Good/Image_processing",
        "images_dir": PROJECT_ROOT / "Good/Image_processing",
        "description": "Task 4 - Good images"
    },
    # ... more sources
]
CLASS_NAMES = {0: "Good", 1: "Flat_line", 2: "Unwash"}
TRAIN_RATIO = 0.70    # 70% for training
VAL_RATIO = 0.15      # 15% for validation
TEST_RATIO = 0.15     # 15% for testing
```

**Why multiple sources?** CVAT exports annotations per-task. Each task may have images from different directories. This script maps each annotation file to its image.

### `merge_dataset` — Pairing Images with Labels
```python
for label_file in label_files:
    stem = label_file.stem               # e.g., "IMG_001" (filename without extension)
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = images_dir / (stem + ext)   # Try to find matching image
        if candidate.exists():
            image_file = candidate
            break
```
**Logic**: For each `.txt` label file, find the image with the same name but with an image extension. Copy both to `dataset/raw/`.

### `split_dataset` — Creating Train/Val/Test
```python
random.seed(RANDOM_SEED)           # Seed = 42 → same split every time
random.shuffle(image_files)        # Randomize order
n_train = int(n_total * 0.70)      # First 70% → train
n_val = int(n_total * 0.15)        # Next 15% → val
# Remaining → test
```
**Why seed?** Reproducibility. Everyone gets the exact same train/val/test split.

---

## 5. `split_dataset.py`

**Purpose**: Alternative standalone script for splitting any directory into train/val/test sets using scikit-learn's stratified splitting.

**Key difference from `prepare_dataset.py`**: This uses `sklearn.model_selection.train_test_split` for **stratified splitting** — it ensures each class has proportional representation in every split. `prepare_dataset.py` does simple random splitting.

---

## 6. `augment_dataset.py`

**Purpose**: Generates augmented copies of training images to increase dataset size. Critical when you have limited data (~87 images).

### YOLO Label Format
```python
# Each line in a .txt label file:
# class_id  x_center  y_center  width  height
# 1         0.453     0.312     0.120  0.085
# ↑ class   ↑ center X ↑ center Y ↑ box width ↑ box height
# All values are normalized (0.0 to 1.0) relative to image size
```

### Augmentation Pipeline (albumentations)
```python
pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),                    # Flip left-right, 50% chance
        A.Rotate(limit=15, p=0.5),                   # Rotate ±15 degrees
        A.RandomBrightnessContrast(p=0.6),           # Change brightness/contrast
        A.GaussNoise(var_limit=(10, 30), p=0.4),     # Add sensor noise
        A.GaussianBlur(blur_limit=(3, 5), p=0.4),    # Slight blur
        A.CLAHE(clip_limit=3.0, p=0.3),              # Adaptive histogram equalization
        A.RandomShadow(p=0.2),                       # Simulate shadows
        A.HueSaturationValue(p=0.3),                 # Color variation
    ],
    bbox_params=A.BboxParams(
        format='yolo',              # ← Tells albumentations our bbox format
        label_fields=['class_ids'], # ← Preserves class labels through augmentation
        min_visibility=0.3,         # ← Drop boxes if <30% visible after transform
    )
)
```

**Why `bbox_params`?** When you rotate or crop an image, the bounding boxes must transform with it. Albumentations does this automatically when you specify `format='yolo'`.

**Key concept — `min_visibility=0.3`**: If an augmentation (e.g., rotation) pushes a defect mostly off-screen, the bounding box is dropped. This prevents training on nearly-invisible defects.

### Three Intensity Levels
| Level | Rotation | Noise | Shadow | Use When |
|-------|----------|-------|--------|----------|
| `light` | ±0° | Low | No | Large dataset (>500 images) |
| `medium` | ±15° | Medium | Yes | Medium dataset (100–500) |
| `heavy` | ±30° | High | Yes | Small dataset (<100) — your case |

### Preview Mode
```python
python augment_dataset.py --preview   # Shows augmented images in OpenCV window, saves nothing
```
**Why?** Always visually check augmentations before committing. Bad augmentations (too aggressive) can hurt training.

---

## 7. `train_model.py`

**Purpose**: Trains the YOLOv11m model on your defect detection dataset.

### `check_environment` — GPU Check (Lines 20–47)
```python
print(f"CUDA Available: {torch.cuda.is_available()}")   # True = GPU ready
print(f"GPU: {torch.cuda.get_device_name(0)}")          # e.g., "NVIDIA GeForce RTX 3060"
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```
**Why?** Always verify GPU before training. CPU training is 10–50x slower.

### `train_yolov26m` — Core Training Function (Lines 50–162)
```python
model = YOLO(weights)    # Load pre-trained YOLOv11m (weights="yolo26m.pt")
# ↑ "Transfer learning" — start from COCO pre-trained weights, not scratch
```

#### Training Parameters Explained
```python
results = model.train(
    data=data_config,          # Path to dataset.yaml (tells YOLO where images/labels are)
    epochs=epochs,             # Number of full passes through the training data
    patience=patience,         # Early stopping: stop if no improvement for N epochs
    batch=batch_size,          # Process N images at once (higher = faster but more GPU RAM)
    imgsz=img_size,            # Resize all images to this size (640×640)

    # --- Optimizer (how the model learns) ---
    optimizer='auto',          # YOLO picks SGD or AdamW based on data
    lr0=0.01,                  # Initial learning rate (how fast the model adjusts)
    lrf=0.01,                  # Final LR = lr0 × lrf (LR decreases over time)
    momentum=0.937,            # SGD momentum (helps escape local minima)
    weight_decay=0.0005,       # L2 regularization (prevents overfitting)
    warmup_epochs=3.0,         # Slowly increase LR for first 3 epochs (stability)

    # --- Augmentation (make data look different each epoch) ---
    degrees=15.0,              # Random rotation ±15°
    translate=0.1,             # Random shift ±10% of image size
    scale=0.3,                 # Random zoom ±30%
    fliplr=0.5,                # Horizontal flip 50% of the time
    mosaic=1.0,                # Mosaic: combine 4 images into one (100% probability)
    mixup=0.1,                 # MixUp: blend 2 images (10% probability)
    hsv_h=0.015,               # Random hue shift (color)
    hsv_s=0.4,                 # Random saturation shift
    hsv_v=0.4,                 # Random brightness shift

    # --- Output ---
    project=project,           # Save results to this folder
    name=name,                 # Experiment name (auto-generated with timestamp)
    amp=True,                  # Automatic Mixed Precision → faster training on GPU
    plots=True,                # Generate training plots (loss curves, PR curves, etc.)
)
```

**Key concepts**:
- **Transfer learning**: Starting from `yolo26m.pt` (pre-trained on COCO, 80 classes) lets the model already "know" what objects look like. You're just teaching it your 3 specific classes.
- **Early stopping**: If validation mAP doesn't improve for `patience` epochs, training stops to prevent overfitting.
- **Mosaic augmentation**: The most powerful YOLO augmentation — combines 4 training images into one, forcing the model to detect objects at different scales and positions.
- **AMP (Mixed Precision)**: Uses 16-bit floats where possible → ~2x faster training, same accuracy.

### `validate_model` (Lines 165–211)
```python
metrics = model.val(split='test', plots=True, save_json=True)
print(f"mAP50:     {metrics.box.map50:.4f}")     # Mean Average Precision @ IoU 0.5
print(f"mAP50-95:  {metrics.box.map:.4f}")        # Mean AP across IoU thresholds 0.5–0.95
```
**mAP explained**: 
- **mAP@50**: "Did you find the defect and got >50% overlap with the ground truth box?" Higher = better.
- **mAP@50-95**: Average across stricter thresholds (50%, 55%, 60%, ... 95%). Much harder to score high on.

---

## 8. `evaluate_model.py`

**Purpose**: Generates a comprehensive evaluation report — metrics, confusion matrix, confidence analysis, speed benchmark.

### `ModelEvaluator` Class — The 4-Step Process

```
Step 1: Run YOLO validation → mAP, Precision, Recall, F1 (per-class + overall)
Step 2: Generate confusion matrix → Which classes get confused?
Step 3: Confidence histogram → Are your detections confident or uncertain?
Step 4: Speed benchmark → How fast is inference on your hardware?
```

### Step 1: Validation Metrics
```python
metrics = self.model.val(data=..., split='test', conf=0.5, iou=0.45)
# F1 is calculated as:
f1 = 2 * precision * recall / (precision + recall)
```
- **Precision**: Of all detections the model made, what % were actually correct?
- **Recall**: Of all real defects in the images, what % did the model find?
- **F1**: Harmonic mean of precision and recall — a single score balancing both.

### Step 3: Confidence Histogram
```python
results = self.model.predict(img_path, conf=0.01)   # Use very low threshold
# ↑ conf=0.01 captures ALL detections, even uncertain ones
# This lets us see the full distribution of confidence scores
```
**Why?** If your defects cluster around 0.3–0.4 confidence, your 0.5 threshold is too high and you're missing real defects. The histogram helps you pick the right threshold.

### Step 4: Speed Benchmark
```python
dummy_image = np.random.randint(0, 255, (640, 640, 3))   # Random image
# Warmup (10 iterations) — first runs are always slower (model loading, GPU warmup)
# Benchmark (100 iterations) — measure consistent speed
fps = 1000 / mean_time_ms    # Convert ms/frame → frames per second
```
**Why?** Your production line has a cycle time (e.g., 1 product every 2 seconds). If inference takes 500ms, you can handle 2 FPS — enough for 1 product every 2 seconds.

### Output
The report saves to `logs/evaluation_report_<timestamp>/`:
- `evaluation_report.json` — Machine-readable full data
- `evaluation_report.md` — Human-readable summary
- `confidence_histogram.png` — Plot of score distributions
- `val_results/confusion_matrix.png` — Class confusion heatmap

---

## 9. `compare_runs.py`

**Purpose**: Compares metrics across multiple training runs to find the best model.

### How It Finds Runs
```python
for csv_file in runs_path.rglob("results.csv"):   # Recursively find every results.csv
    # Ultralytics auto-saves results.csv in each training run directory
    # Contains per-epoch: loss, mAP, precision, recall, learning rate
```

### Parsing `results.csv`
```python
# Ultralytics results.csv has columns like:
# epoch, train/box_loss, train/cls_loss, val/box_loss, metrics/mAP50(B), metrics/mAP50-95(B), ...
best_epoch = max(epochs, key=lambda e: e.get('mAP50-95', 0))
# ↑ Find which epoch had the highest mAP50-95
```

### Comparison Table Output
```
Run Name                        Epochs  mAP50   mAP50-95  Prec    Recall  Date
─────────────────────────────────────────────────────────────────────────────
defect_gpu_v3                      100  0.8234    0.6121  0.7845  0.8012  2026-02-16
pipeline_small_dataset_20260304    300  0.9012    0.7234  0.8567  0.8901  2026-03-04
🏆 Best run: pipeline_small_dataset_20260304 (mAP50-95: 0.7234)
```

### Overlay Plots
Generates 6 plots overlaid for all runs:
- mAP@50, mAP@50-95 — Did accuracy improve?
- Precision, Recall — Detection confidence vs. coverage?
- Train/Val Box Loss — Is the model still learning or overfitting?

---

## 10. `defect_detector.py`

**Purpose**: Production inference — runs defect detection on live camera feeds or individual images.

### Three Main Classes

#### `DefectDetector` — Core Detection
```python
def detect(self, image: np.ndarray) -> dict:
    results = self.model.predict(image, conf=self.conf_threshold, iou=self.iou_threshold)
    # ↑ Run YOLO inference on one image
    
    for box in results[0].boxes:
        class_id = int(box.cls[0])              # Which class (0=Good, 1=Flat_line, 2=Unwash)
        confidence = float(box.conf[0])         # How sure the model is (0.0 to 1.0)
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates (pixel values)
    
    is_defective = any(d['class'] != 'Good' for d in detections)
    # ↑ If ANY detection is not "Good", the product fails inspection
```

**Key concept — NMS (Non-Maximum Suppression)**: When the model detects the same defect multiple times with overlapping boxes, NMS keeps only the most confident one. Controlled by `iou_threshold=0.45`.

#### `InspectionLogger` — SQLite Database Logging
```python
conn = sqlite3.connect(self.db_path)          # Connect to local database
cursor.execute("""CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,      # Auto-incrementing ID
    timestamp TEXT,                             # When the inspection happened
    is_defective INTEGER,                       # 0 = pass, 1 = fail
    num_defects INTEGER,                        # How many defects found
    defect_classes TEXT,                        # JSON list of defect names
    confidence_scores TEXT,                    # JSON list of confidences
    ...
)""")
```
**Why SQLite?** Lightweight, no server needed, works offline. Perfect for factory floor use.

#### `ProductionInspectionSystem` — Full System
```python
def inspect_once(self):
    image = self.camera.capture()              # 1. Get image from camera
    result = self.detector.detect(image)        # 2. Run YOLO detection
    self.logger.log(result, image_path)         # 3. Log to database
    if result['is_defective']:
        # Trigger alert (buzzer, PLC signal, etc.)
```
This class combines camera + detector + logger into one pipeline for continuous production inspection.

---

## 11. `export_model.py`

**Purpose**: Converts trained `.pt` model to optimized formats for production deployment.

### Export Formats
```python
model.export(format='onnx')          # ONNX — universal format, runs anywhere
model.export(format='engine')        # TensorRT — NVIDIA GPU optimized (fastest)
model.export(format='torchscript')   # TorchScript — PyTorch's serialization format
model.export(format='openvino')      # OpenVINO — Intel CPU/GPU optimized
```

**When to use what?**
| Format | Best For | Speed |
|--------|----------|-------|
| `.pt` (PyTorch) | Development & training | Baseline |
| `.onnx` | Cross-platform deployment | ~1.5x faster |
| `.engine` (TensorRT) | NVIDIA GPU production | ~3–5x faster |
| `.xml` (OpenVINO) | Intel CPU/iGPU | ~2x faster |

### `benchmark_models` — Speed Comparison
```python
# Warmup (5 iterations) then benchmark (50 iterations)
# Outputs: file size (MB), inference time (ms), FPS
```
Use `--benchmark` flag to compare your `.pt` vs exported model speed:
```bash
python export_model.py models/best.pt --format onnx --benchmark
```

---

## 12. `webcam_demo.py`

**Purpose**: Minimal script to run live inference on your webcam — great for demos.

```python
model = YOLO(model_path)                              # Load trained model
cap = cv2.VideoCapture(0)                              # Open default webcam (index 0)

while True:
    ret, frame = cap.read()                            # Read one frame from webcam
    results = model(frame, verbose=False)              # Run YOLO inference
    annotated_frame = results[0].plot()                # Draw bounding boxes on frame
    cv2.imshow("YOLOv11 Live Inference", annotated_frame)  # Display in window

    if cv2.waitKey(1) & 0xFF == ord('q'):              # Press 'q' to quit
        break

cap.release()                                          # Release webcam
cv2.destroyAllWindows()                                # Close all OpenCV windows
```

**Line-by-line**:
- `cv2.VideoCapture(0)` — Opens the first camera (0 = default webcam, 1 = second camera)
- `model(frame, verbose=False)` — Runs detection silently (no print output)
- `results[0].plot()` — Ultralytics draws colorful bounding boxes with class names and confidence scores
- `cv2.waitKey(1) & 0xFF == ord('q')` — Wait 1ms for keypress; `& 0xFF` handles cross-platform key encoding

---

## 13. `run_pipeline.py`

**Purpose**: **Master script** that chains all other scripts into an automated pipeline.

### How It Works
```python
# Adds scripts/ directory to Python's import path
sys.path.insert(0, scripts_dir)

# Then imports functions directly from other scripts:
from validate_images import ImageQualityValidator
from augment_dataset import augment_dataset
from train_model import check_environment, train_yolov26m
from evaluate_model import ModelEvaluator
from compare_runs import discover_runs, print_comparison_table
```

### Pipeline Modes

```python
mode_map = {
    'full':          run_full_pipeline,     # Validate → Augment → Train → Evaluate → Compare
    'train-eval':    run_train_eval,        # Train → Evaluate → Compare
    'eval-only':     run_eval_only,         # Evaluate existing model
    'augment-only':  run_augment_only,      # Augment dataset only
}
```

### Hyperparameter Preset Loading
```python
def load_preset(preset_name):
    import yaml
    with open('config/hyperparams.yaml', 'r') as f:
        all_presets = yaml.safe_load(f)              # Load all presets
    preset = all_presets[preset_name]                  # Pick the requested one
    # Returns dict: {'epochs': 300, 'batch_size': 8, 'lr0': 0.001, ...}
```
This is how `--preset small_dataset` translates to actual training parameters.

### Full Pipeline Flow
```
step_validate()  →  Check image quality, report issues
step_augment()   →  Generate augmented copies (5x by default)
step_train()     →  Load preset → Train YOLOv11m → Save best.pt
step_evaluate()  →  Full report: mAP, confusion matrix, FPS
step_compare()   →  Compare with all previous runs
step_export()    →  (Optional) Export to ONNX
```

---

## 14. Config Files

### `config/dataset.yaml`
```yaml
path: C:/.../dataset       # Root path to your dataset
train: images/train         # Subdirectory for training images
val: images/val             # Subdirectory for validation images
test: images/test           # Subdirectory for test images

names:
  0: Good                   # Class 0 = no defect
  1: Flat_line              # Class 1 = flat-line defect
  2: Unwash                 # Class 2 = unwash defect

nc: 3                       # Number of classes (must match len(names))
```
**This is the single most important config** — YOLO reads it to know where your data is and what classes exist.

### `config/system_config.json`
```json
{
    "camera": {
        "exposure_time_us": 15000,    // 15ms exposure — controls brightness
        "gain_db": 0,                 // No gain — cleaner image, less noise
        "pixel_format": "Mono8"       // Grayscale 8-bit — sufficient for defect detection
    },
    "model": {
        "confidence_threshold": 0.5,  // Only show detections with >50% confidence
        "iou_threshold": 0.45,        // NMS: suppress overlapping boxes with IoU > 0.45
        "image_size": 640             // Resize input to 640×640 for inference
    },
    "inspection": {
        "save_images": true,          // Keep images of inspected products
        "log_to_database": true,      // Write results to SQLite
        "database_path": "logs/inspections.db"
    }
}
```

### `config/hyperparams.yaml`
Contains 4 presets. Each one has all training parameters with comments explaining **when to use it** and **why**:
- `baseline` — default values, use as control experiment
- `small_dataset` — tuned for <200 images: lower LR, more augmentation, longer training
- `fine_detail` — 1280px input for small/subtle defects
- `fast_training` — 20 epochs for quick sanity checks

---

## 🔑 Key Concepts Glossary

| Term | Meaning |
|------|---------|
| **mAP** | Mean Average Precision — primary object detection metric |
| **IoU** | Intersection over Union — overlap between predicted and real bounding box |
| **NMS** | Non-Maximum Suppression — removes duplicate detections |
| **Transfer Learning** | Starting from a pre-trained model instead of training from scratch |
| **Early Stopping** | Stop training when validation metric stops improving |
| **AMP** | Automatic Mixed Precision — uses FP16 where safe for faster training |
| **Mosaic** | Combines 4 images into one for augmentation |
| **Epoch** | One complete pass through the entire training dataset |
| **Batch Size** | Number of images processed simultaneously per training step |
| **Learning Rate** | How much the model adjusts its weights per step (smaller = more stable) |
| **Patience** | How many epochs without improvement before early stopping activates |
| **YOLO Format** | Label format: `class_id x_center y_center width height` (all normalized) |
| **ONNX** | Open Neural Network Exchange — universal model format |
| **TensorRT** | NVIDIA's inference optimization engine |
| **Confusion Matrix** | Grid showing which classes get confused with each other |
| **F1 Score** | Harmonic mean of Precision and Recall |

---

*This document is maintained as a learning reference. Update it when scripts change significantly.*

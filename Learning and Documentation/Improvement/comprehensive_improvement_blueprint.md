# 🔬 Comprehensive Software Improvement Blueprint

> Based on full review of all 10 scripts, 5 configs, dataset structure, and conversation history.

---

## Table of Contents
1. [Critical Bugs to Fix Now](#1--critical-bugs-to-fix-now)
2. [Inference & Real-Time Performance](#2--inference--real-time-performance)
3. [Machine Learning & Model Quality](#3--machine-learning--model-quality)
4. [Image Preprocessing & Computer Vision](#4--image-preprocessing--computer-vision)
5. [Data Pipeline & Dataset Management](#5--data-pipeline--dataset-management)
6. [System Architecture & Code Quality](#6--system-architecture--code-quality)
7. [Dashboard & User Interface](#7--dashboard--user-interface)
8. [Industrial Integration & Deployment](#8--industrial-integration--deployment)
9. [Testing & Reliability](#9--testing--reliability)
10. [Tools & Libraries to Add](#10--tools--libraries-to-add)

---

## 1. 🚨 Critical Bugs to Fix Now

These are issues I found in the current codebase that should be fixed immediately.

### 1.1 Config: Confidence Threshold = 3.0
**File:** `config/system_config.json` line 11
```json
"confidence_threshold": 3.0
```
This value should be between 0.0 and 1.0 (e.g., `0.03` or `0.30`). A value of `3.0` is invalid and would break any code that reads it literally.

### 1.2 Dataset Class Mismatch
**File:** `config/dataset.yaml` defines **1 class** (`Scratch`, nc=1), but `config/system_config.json` lists **3 classes** (`Good`, `Flat_line`, `Unwash`). Your detector currently only trains on `Scratch`. If you have images of other defect types annotated, they are being silently ignored during training. Reconcile these two files.

### 1.3 Bare `except:` Clauses
**File:** `defect_detector.py` lines 341, 364–365, and `dashboard.py` line 90 have bare `except:` or `except: pass` blocks. These silently swallow critical errors (e.g., GPU OOM, corrupt database). At minimum, change to `except Exception as e: print(f"Error: {e}")`.

---

## 2. ⚡ Inference & Real-Time Performance

### 2.1 TensorRT Export for Production
**What:** Export your `.pt` model to TensorRT `.engine` format.
**Why:** PyTorch `.pt` format is designed for training flexibility, not speed. TensorRT compiles the model into fused CUDA kernels optimized for your exact GPU. Typical speedup: **2×–5× faster inference**.
**How:** Your `export_model.py` already supports this. Run:
```bash
python export_model.py best.pt --format engine --half --imgsz 1280
```
Then modify `defect_detector.py` to load the `.engine` file instead of `.pt`. The Ultralytics YOLO class handles this automatically — `YOLO("best.engine")` just works.

### 2.2 ONNX Runtime as a Fallback
**What:** Export to ONNX and use `onnxruntime-gpu` for inference.
**Why:** ONNX is portable across hardware vendors (NVIDIA, AMD, Intel, ARM). If you ever deploy to a machine without a TensorRT-compatible GPU, ONNX Runtime is your safety net. It also supports quantization to INT8 for even faster inference on CPU.
**Tool:** `pip install onnxruntime-gpu`

### 2.3 Multithreaded Producer-Consumer Pipeline
**What:** Separate camera capture, YOLO inference, and display into 3 threads connected by `queue.Queue`.
**Why:** Currently in `start()`, the camera blocks while YOLO runs, and YOLO blocks while `cv2.imshow` draws. Your FPS is capped by the slowest step.
**Implementation:**
```
Thread 1 (Camera):   capture() → put frame in Queue A
Thread 2 (Detector): get frame from Queue A → detect() → put result in Queue B
Thread 3 (Display):  get result from Queue B → draw_results() → imshow()
```
Use `queue.Queue(maxsize=2)` to prevent memory buildup. This is the single biggest performance improvement you can make.

### 2.4 Frame Skipping (Drop Stale Frames)
**What:** If the detector is slower than the camera, drop old frames and always process the latest one.
**Why:** Processing frame #5 when frame #15 is already available means you're showing the operator stale information. With `pylon.GrabStrategy_LatestImageOnly` you already do this at the hardware level, but add it at the software queue level too.
**How:** In your consumer thread, drain the queue and only process the last frame:
```python
while not q.empty():
    frame = q.get()  # discard all but the final one
```

### 2.5 GPU Preprocessing with OpenCV CUDA or `torchvision.transforms`
**What:** Move image resize and greyscale conversion to the GPU.
**Why:** `cv2.resize(image, (1280, 1280))` on a 5MP Basler image is CPU-bound. If you move this to GPU via `cv2.cuda.resize()` or by using PyTorch tensors directly, you eliminate a CPU bottleneck.
**Tool:** Build OpenCV with CUDA support, or use `kornia` (a differentiable computer vision library on PyTorch).

### 2.6 Batch Inference for Video Mode
**What:** In `detect_from_video()`, buffer N frames and run inference in a single batch.
**Why:** GPU utilization is poor when processing one frame at a time. With `batch_size=4`, the GPU processes 4 frames simultaneously, improving throughput by ~3×.
**How:** Accumulate 4 frames, then call `self.model(batch_of_4_frames, ...)`.

---

## 3. 🧠 Machine Learning & Model Quality

### 3.1 Multi-Class Expansion
**What:** Expand beyond the single `Scratch` class.
**Why:** Your `system_config.json` already envisions `Good`, `Flat_line`, and `Unwash`. A multi-class model provides richer defect categorization and enables per-class quality control rules.
**Action:** Annotate existing images with all classes in CVAT, update `dataset.yaml` to `nc: 4`, retrain.

### 3.2 Anomaly Detection (Unsupervised Add-On)
**What:** Train a separate anomaly detection model that learns what a "good" part looks like and flags anything unusual.
**Why:** YOLO only detects defects it was trained on. A brand-new defect type (contamination, discoloration, foreign object) will be invisible to YOLO. Anomaly detection catches the *unknown unknowns*.
**Tools:**
- **Anomalib** (Intel, open-source) — Implements PatchCore, PaDiM, FastFlow, EfficientAD
- **Algorithm recommendation:** Start with **PatchCore** — it's fast, simple, and works well with small "good" datasets
**Architecture:** Run YOLO and PatchCore in parallel. YOLO classifies known defects; PatchCore scores overall abnormality. If PatchCore score > threshold AND YOLO found nothing, flag as "Unknown Anomaly".

### 3.3 Ensemble / Model Fusion 
**What:** Run 2 different models (e.g., YOLOv11m + YOLOv11s, or detection + segmentation) and fuse their predictions.
**Why:** Two models with different architectures make different mistakes. Fusing their outputs (e.g., Weighted Boxes Fusion) reduces both false positives and false negatives.
**Tool:** `pip install ensemble-boxes` — implements WBF, NMS, Soft-NMS, NMW.

### 3.4 Object Tracking for Part Deduplication   (Not needed)
**What:** Use ByteTrack or BoT-SORT to track defects across frames.
**Why:** In live mode, the same physical defect is detected in every frame as the part moves under the camera. Without tracking, your logger records it as 100 separate defects. With tracking, it's counted exactly once.
**How:** Ultralytics has built-in tracking:
```python
results = model.track(frame, tracker="bytetrack.yaml", persist=True)
```
Each defect gets a unique `track_id`. Log only when a new `track_id` appears.

### 3.5 Test-Time Augmentation (TTA)
**What:** Run inference multiple times on flipped/scaled versions of the same image and merge results.
**Why:** TTA can significantly boost mAP (especially on small/subtle defects) at the cost of 3-5× slower inference. Use it for batch evaluation of critical parts, not for real-time.
**How:** `model.predict(image, augment=True)` — Ultralytics supports this natively.

### 3.6 Knowledge Distillation (Faster Small Model)
**What:** Train a lightweight YOLOv11n (nano) or YOLOv11s (small) student model that mimics your large YOLOv11m teacher.
**Why:** Smaller model = faster inference = higher FPS = better for production. The student can achieve 90-95% of the teacher's accuracy at 3× the speed.
**How:** Ultralytics doesn't have built-in distillation, but you can use the teacher model to generate pseudo-labels on unlabeled data and train the student on those.

### 3.7 Confidence Calibration
**What:** Apply Platt scaling or temperature scaling to calibrate model confidence scores.
**Why:** A YOLO model saying "80% confident" doesn't always mean there's an 80% chance of a defect. Calibration aligns the numerical confidence with actual accuracy, making your threshold tuning more meaningful.
**Tool:** `pip install netcal` or implement temperature scaling manually on validation set outputs.

### 3.8 Active Learning Pipeline
**What:** Automatically save "uncertain" detections (confidence between 0.05 and 0.30) to a review folder.
**Why:** These edge cases are exactly the images that will improve your model the most when annotated and added to the training set. Without this, they're lost forever.
**Implementation:** In `InspectionLogger.log()`, add:
```python
if 0.05 < max_conf < 0.30:
    shutil.copy(image_path, "dataset/needs_review/")
```
Then periodically review these in CVAT and retrain.

### 3.9 Hyperparameter Tuning with Ray Tune
**What:** Automated hyperparameter search instead of manual preset tuning.
**Why:** Your `hyperparams.yaml` has 5 hand-crafted presets. Automated search can explore hundreds of combinations and find configurations you'd never try manually.
**How:** Already documented in your `hyperparams.yaml` comments:
```python
model.tune(data="config/dataset.yaml", epochs=30, iterations=50)
```

### 3.10 Class-Weighted Loss / Focal Loss
**What:** Apply higher loss weights to underrepresented classes.
**Why:** If 80% of your annotations are "Scratch" and 20% are "Good", the model learns to always predict Scratch. Focal loss or class weights force balanced attention.
**How:** In `hyperparams.yaml`, adjust `cls` weight. For focal loss, set `fl_gamma` in the Ultralytics training config (e.g., `fl_gamma: 1.5`).

---

## 4. 🖼️ Image Preprocessing & Computer Vision

### 4.1 Dynamic Exposure / White Balance Compensation
**What:** Detect and compensate for lighting changes in real-time.
**Why:** Factory lighting changes throughout the day (natural light, shift changes, bulb aging). A brightness shift can cause the model to miss defects or hallucinate them.
**Implementation:** Calculate the mean brightness of each frame. If it drifts beyond ±15% of a calibrated baseline, apply histogram matching to normalize it back. Or adjust the Basler camera's exposure programmatically via `pypylon`.

### 4.2 Image Quality Gate
**What:** Before running YOLO, check if the image is usable (not blurry, not over/underexposed, part is present).
**Why:** Running inference on a blurry or empty frame wastes GPU time and produces false results.
**Checks to add:**
- **Blur detection:** Laplacian variance < threshold → skip frame, show "FOCUS ERROR" on HUD
- **Exposure check:** Mean pixel value < 30 or > 230 → skip, show "LIGHTING ERROR"
- **Part presence:** Contour area < minimum → skip, show "NO PART DETECTED"

### 4.3 Perspective / Lens Distortion Correction
**What:** Apply camera calibration to undistort images before inference.
**Why:** Basler cameras with wide-angle lenses introduce barrel distortion, especially at the edges. A defect near the image border may be geometrically warped, reducing detection accuracy.
**How:** Use OpenCV's `cv2.calibrateCamera()` with a checkerboard pattern, then `cv2.undistort()` on every frame.

### 4.4 Background Subtraction for Moving Parts
**What:** If parts move on a conveyor, use MOG2 or KNN background subtraction to isolate the part from the belt.
**Why:** Your current dynamic margin logic uses edge detection + contours to find the part boundary. Background subtraction is more robust when the background (conveyor belt) is consistent.
**How:** `cv2.createBackgroundSubtractorMOG2()` — train it on empty belt frames, then subtract.

### 4.5 Multi-Scale Inference
**What:** Run the model at multiple resolutions (e.g., 640, 1024, 1280) and merge results.
**Why:** Large defects are better detected at low resolution (more context). Small defects are better detected at high resolution (more pixels). Multi-scale combines both.
**How:** Run `model(image, imgsz=640)` and `model(image, imgsz=1280)`, then merge with WBF.

### 4.6 Color Space Experiments
**What:** Try HSV, LAB, or YCrCb color spaces instead of / in addition to your rust-aware greyscale.
**Why:** In LAB color space, the `a` channel directly encodes red-green intensity — potentially more discriminative for rust detection than your RGB ratio formula.
**Action:** Experiment in `convert_to_greyscale()` by adding a flag to switch between preprocessing strategies during training and inference.

---

## 5. 📦 Data Pipeline & Dataset Management

### 5.1 Expand Dataset Size (Target: 1000+ Images)
**What:** Your dataset has ~200 images. Aim for at least 500-1000 per class for robust detection.
**Why:** YOLO performance improves roughly logarithmically with dataset size. Going from 200 → 1000 images can improve mAP by 10-20%.
**Methods:**
- Capture more images with `camera_capture.py`
- Use your `augment_dataset.py` with `--multiplier 5`
- **Synthetic data generation** — use image editing or generative AI to create artificial defects on "Good" images

### 5.2 Smarter Augmentations (Domain-Specific)
**What:** Add factory-specific augmentations to `augment_dataset.py`.
**Why:** Your current pipeline uses generic transforms. Factory environments have specific visual variations.
**Add these Albumentations transforms:**
```python
A.RandomShadow(p=0.3)            # Simulates conveyor shadows
A.RandomFog(p=0.2)               # Simulates oil mist / dust
A.RandomBrightnessContrast(       # Simulates lighting variation
    brightness_limit=0.3, contrast_limit=0.3, p=0.5)
A.MotionBlur(blur_limit=5, p=0.2) # Simulates vibration
A.ImageCompression(p=0.1)         # Simulates camera compression artifacts
A.CoarseDropout(p=0.2)           # Simulates occlusion (debris on lens)
```

### 5.3 Copy-Paste Augmentation for Defects
**What:** Cut defect regions from one image and paste them onto clean parts.
**Why:** This is one of the most effective augmentations for object detection with small datasets. It creates realistic training examples with controlled defect placement.
**Tool:** Built into Ultralytics (`copy_paste: 0.3` in hyperparams). Also implementable in `augment_dataset.py` with masks.

### 5.4 Dataset Version Control with DVC
**What:** Track your dataset versions alongside code versions.
**Why:** When you retrain, you need to know exactly which images were in the training set. If you add new images and performance drops, you can't debug without version history.
**Tool:** `pip install dvc` — DVC (Data Version Control) integrates with Git but handles large binary files (images) efficiently.

### 5.5 Auto-Annotation Pipeline
**What:** Use your trained model to pre-annotate new images, then human-review in CVAT.
**Why:** Manual annotation from scratch takes 2-5 minutes per image. Pre-annotation + correction takes 30 seconds.
**How:** Run `model.predict()` on new images, convert output to CVAT XML format, import into CVAT for review.

### 5.6 Data Integrity Validation
**What:** Add automated checks for dataset health before training.
**Why:** Corrupt images, empty label files, class ID mismatches, and label/image filename misalignment cause silent training failures.
**Checks to automate in `validate_images.py`:**
- Every image has a corresponding `.txt` label file
- Every label file class ID is within valid range (0 to nc-1)
- No duplicate images (hash comparison)
- No zero-byte images
- All polygon coordinates are within [0, 1]
- Class distribution report (flag if any class < 10% of total)

---

## 6. 🏗️ System Architecture & Code Quality

### 6.1 Configuration Validation with Pydantic
**What:** Replace raw `json.load()` / `yaml.safe_load()` with Pydantic models.
**Why:** Catches config errors at startup instead of at runtime. For example, your `confidence_threshold: 3.0` bug would throw a clear error: `"confidence_threshold must be between 0.0 and 1.0"`.
**Example:**
```python
from pydantic import BaseModel, Field

class InspectionConfig(BaseModel):
    confidence_threshold: float = Field(ge=0.0, le=1.0, default=0.03)
    iou_threshold: float = Field(ge=0.0, le=1.0, default=0.45)
    min_defect_area_px: float = Field(ge=0, default=150.0)
```

### 6.2 Structured Logging with `loguru`
**What:** Replace `print()` statements with proper logging.
**Why:** 
- Log levels (DEBUG, INFO, WARNING, ERROR) let you filter noise
- Automatic timestamps
- File rotation (don't fill the disk)
- Colored console output
**Tool:** `pip install loguru`

### 6.3 Proper Error Recovery & Watchdog
**What:** Add automatic restart logic when the camera disconnects or YOLO crashes.
**Why:** In production, the system must run 24/7. A single exception shouldn't kill the process. Add:
- Camera reconnection with exponential backoff
- GPU OOM recovery (reduce batch size, retry)
- Watchdog timer — if no frame is processed for 30 seconds, force-restart the pipeline

### 6.4 Model Registry & Versioning
**What:** Track which model version is deployed and its performance metrics.
**Why:** When you deploy `best.pt`, you lose track of which training run produced it. Add a system that records:
- Model hash (SHA256 of the `.pt` file)
- Training run ID
- mAP/precision/recall at deployment time
- Deployment date
**Simple approach:** A `models/registry.json` file updated by `export_model.py`.

### 6.5 Plugin Architecture for Preprocessing
**What:** Make the preprocessing pipeline configurable and swappable.
**Why:** Your greyscale + CLAHE + rust darkening pipeline is hardcoded. Different products might need different preprocessing. Allow:
```yaml
preprocessing:
  - type: clahe
    clip_limit: 3.0
  - type: rust_darkening
    strength: 0.55
  - type: resize
    size: [1280, 1280]
```

### 6.6 Remove Code Duplication
**What:** The settings window toggle logic (trackbar creation + reading) is copy-pasted 4 times in `defect_detector.py` (live mode, video mode, image from file, image from camera).
**Why:** Bug fixes need to be applied in 4 places. Extract into a `SettingsWindow` class.

### 6.7 Environment Containerization with Docker
**What:** Package your entire environment (Python, CUDA, dependencies) in a Docker container.
**Why:** Your current setup relies on `defect_env_gpu311` venv with specific CUDA/PyTorch versions. If the system gets reimaged or you move to a new machine, setup could take days. Docker ensures reproducibility.
**How:** Create a `Dockerfile` based on `nvidia/cuda:12.x-cudnn-runtime`.

---

## 7. 🖥️ Dashboard & User Interface

### 7.1 Real-Time Dashboard (WebSocket Live Feed)
**What:** Stream inference results to the Streamlit dashboard in real-time.
**Why:** Currently, the dashboard only shows historical data from SQLite. Operators have to look at the OpenCV window AND the browser. Merge them.
**How:** Use FastAPI + WebSocket to push live results, or use Streamlit's `st.empty()` with auto-refresh.

### 7.2 Shift/Batch Analytics
**What:** Your `inspections` DB has `shift` and `batch` columns but they're never populated.
**Why:** Enabling these allows you to answer: "Which shift has the highest defect rate?" and "Which batch of raw material produces the most scratches?"
**Action:** Add `--shift` and `--batch` CLI args to `defect_detector.py`, pass them to `self.logger.log()`.

### 7.3 SPC Charts (Statistical Process Control)
**What:** Add X-bar, R-chart, and p-chart to the dashboard.
**Why:** These are the industry standard for monitoring manufacturing quality. An X-bar chart shows whether the defect rate is within control limits or drifting.
**Tool:** `pip install pyspc` or implement manually with Plotly.

### 7.4 Defect Heatmap Overlay
**What:** Aggregate all defect bounding boxes across hundreds of inspections and generate a spatial heatmap.
**Why:** Shows which physical region of the part has the most defects. This directly points to tooling wear or machine misalignment.
**Implementation:** For each logged defect, add its normalized bbox center to a 2D histogram. Render with `matplotlib.pyplot.imshow()` or `plotly.express.density_heatmap()`.

### 7.5 Trend Alerting & Notifications
**What:** Trigger alerts when defect rate exceeds threshold.
**Why:** An operator might not notice a gradual increase in defects. The system should proactively alert.
**Implementation ideas:**
- Email alerts via `smtplib`
- Slack/Teams webhooks
- Physical tower light via USB relay
- Sound alarm via `playsound` library
- Rule: "If 5 consecutive NOK results → alert"

### 7.6 Model Performance Drift Monitoring
**What:** Track model confidence statistics over time. If average confidence drops below a threshold, flag "model may need retraining".
**Why:** As manufacturing conditions change (new material, new tooling), the model's accuracy degrades silently.
**How:** Log average confidence per session in the dashboard. Plotly line chart with a red dashed "retraining threshold" line.

### 7.7 Migrate from OpenCV GUI to Web-Based UI
**What:** Replace `cv2.imshow` with a web-based live feed.
**Why:** OpenCV's GUI is fragile (crashes on thread issues, no remote access, no mobile support). A web UI can be accessed from any device on the network.
**Stack:** FastAPI (backend) + MJPEG streaming + React or plain HTML/JS (frontend). FastAPI serves annotated frames as a multipart JPEG stream.

---

## 8. 🏭 Industrial Integration & Deployment

### 8.1 Hardware Trigger (Not Software Polling)
**What:** Connect an optical sensor on the conveyor to the Basler camera's hardware trigger input.
**Why:** Software polling (`GrabStrategy_LatestImageOnly`) captures whenever the software asks, regardless of part position. Hardware triggering fires the camera precisely when the part is at the correct position under the lens. This eliminates your dynamic margin guesswork.

### 8.2 PLC / OPC-UA Communication
**What:** Send inspection results (OK/NOK) to a PLC via OPC-UA protocol.
**Why:** In a real production line, the inspection result must trigger a physical action (e.g., diverter arm rejects the part, conveyor stops). PLC integration is how you connect software to physical actuators.
**Tool:** `pip install opcua` (python-opcua library)

### 8.3 Barcode/QR Integration
**What:** Read a barcode or QR code on each part before inspection.
**Why:** Links each inspection result to a specific part serial number. Enables full traceability: "Part #12345 had a scratch at zone Thread on April 16 at 2:30 PM."
**Tool:** `pip install pyzbar` + `cv2.decode()` — detect and read barcodes from the same camera feed.

### 8.4 Multi-Camera Support
**What:** Inspect the same part from multiple angles simultaneously.
**Why:** A camera looking straight down misses defects on vertical surfaces. Adding a second camera at 45° gives full 3D coverage.
**Implementation:** Extend `BaslerCamera` to support multiple `InstantCamera` instances. Run inference on each view independently, then merge results.

### 8.5 Edge Deployment (Jetson / Industrial PC)
**What:** Optimize for deployment on NVIDIA Jetson Orin or similar edge devices.
**Why:** A full workstation with a desktop GPU is expensive and overkill. Jetson Orin can run TensorRT inference at 30+ FPS for a fraction of the cost and power.
**Action:** Export to TensorRT FP16 engine with `imgsz=640` (smaller for edge). Test on Jetson with `jetson-stats` to monitor GPU/CPU/memory.

### 8.6 Results Transfer to MES/ERP
**What:** Push inspection data to a Manufacturing Execution System or ERP.
**Why:** Quality data lives in your local SQLite database and is invisible to the broader factory IT ecosystem. Integration with MES (e.g., Ignition, Siemens Opcenter) enables organization-wide quality dashboards and compliance reporting.
**Protocol:** REST API (JSON payloads) or MQTT for lightweight messaging.

---

## 9. 🧪 Testing & Reliability

### 9.1 Unit Tests for Core Functions
**What:** Add `pytest` tests for `detect()`, `convert_to_greyscale()`, `read_yolo_labels()`, `parse_results_csv()`.
**Why:** Every time you modify preprocessing, you risk breaking inference. Tests catch regressions before deployment.
**Priority tests:**
```python
def test_greyscale_output_shape():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = convert_to_greyscale(img)
    assert result.shape == (480, 640, 3)  # still BGR

def test_detect_returns_required_keys():
    detector = DefectDetector("best.pt")
    result = detector.detect(dummy_image)
    assert 'is_defective' in result
    assert 'defects' in result
    assert 'inference_time_ms' in result
```

### 9.2 Integration Test: Full Pipeline
**What:** End-to-end test that runs `MockCamera → detect → draw_results → log to DB → read from DB`.
**Why:** Individual units may work but the pipeline can break at connection points (e.g., JSON serialization of defect polygons).

### 9.3 Regression Test Suite for Model Updates
**What:** A fixed set of 20-30 "golden" test images with known expected results.
**Why:** When you retrain the model, you need to immediately verify: "Does the new model still detect the scratch on test_image_7.jpg?" If it doesn't, you've regressed.
**Implementation:** Store golden images + expected class/confidence in `tests/golden/`. Run `evaluate_model.py` against this set and assert mAP > minimum threshold.

### 9.4 Performance Benchmarking CI
**What:** Automatically benchmark inference speed after model export.
**Why:** A model change or dependency update might silently degrade FPS. Catch it automatically.
**How:** In a CI/CD pipeline (GitHub Actions or local Jenkins), run `export_model.py --benchmark` and assert FPS > minimum.

### 9.5 Canary / Shadow Deployment
**What:** Run the new model alongside the old model and compare results without affecting production.
**Why:** Safely validate a new model on live production data before switching over.
**Implementation:** Load both models in `DefectDetector`. Run inference with both. Log new model results to a separate DB table. Alert if the new model disagrees with the old model on > 5% of frames.

---

## 10. 🛠️ Complete Tools & Libraries Reference

| Category | Tool | Purpose | Priority |
|----------|------|---------|----------|
| **Inference** | TensorRT | 2-5× faster GPU inference | 🔴 High |
| **Inference** | ONNX Runtime | Portable cross-platform inference | 🟡 Medium |
| **ML** | Anomalib (PatchCore) | Unsupervised anomaly detection | 🔴 High |
| **ML** | ensemble-boxes (WBF) | Multi-model prediction fusion | 🟡 Medium |
| **ML** | Ray Tune | Automated hyperparameter search | 🟡 Medium |
| **ML** | netcal | Confidence calibration | 🟢 Low |
| **Tracking** | ByteTrack | Object tracking / deduplication | 🔴 High |
| **CV** | kornia | GPU-accelerated image transforms | 🟡 Medium |
| **Data** | DVC | Dataset version control | 🟡 Medium |
| **Data** | Albumentations (extras) | Domain-specific augmentations | 🟡 Medium |
| **Config** | Pydantic | Config validation & typing | 🔴 High |
| **Logging** | loguru | Structured, leveled logging | 🔴 High |
| **Testing** | pytest | Unit & integration testing | 🔴 High |
| **Dashboard** | FastAPI | Web API + MJPEG streaming | 🟡 Medium |
| **Dashboard** | pyspc | Statistical Process Control charts | 🟢 Low |
| **Industrial** | python-opcua | PLC communication | 🟢 Low |
| **Industrial** | pyzbar | Barcode/QR reading | 🟢 Low |
| **Industrial** | MQTT (paho-mqtt) | Lightweight messaging to MES | 🟢 Low |
| **DevOps** | Docker | Environment containerization | 🟡 Medium |
| **Monitoring** | Prometheus + Grafana | System health monitoring | 🟢 Low |

---

## 📋 Suggested Implementation Order

1. **Fix critical bugs** (Section 1) — 30 minutes
2. **Multithreaded pipeline** (2.3) — biggest FPS impact
3. **TensorRT export** (2.1) — biggest inference speed impact
4. **Object tracking** (3.4) — stop duplicate defect logging
5. **Pydantic configs** (6.1) + **loguru logging** (6.2) — prevent future bugs
6. **Unit tests** (9.1) — safety net for all future changes
7. **Active learning pipeline** (3.8) — continuous model improvement
8. **Image quality gate** (4.2) — reduce false positives from bad frames
9. **Anomaly detection** (3.2) — catch unknown defects
10. **Dashboard enhancements** (7.1-7.6) — operational visibility

---

*Pick any item from this list and I can create a detailed implementation plan with code for it.*

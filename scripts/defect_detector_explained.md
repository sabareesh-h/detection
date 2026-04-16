# Defect Detector — Complete Code Walkthrough
**In this project, `defect_detector.py` is the "Brain" of inference.

`camera_capture.py` gives frames, but this file decides PASS/REJECT, draws overlays, and logs inspection outputs.

Here are the 4 main reasons why we need this file:

1. AI Inference Core (Decision Engine)
It loads the trained YOLO model and performs actual defect prediction on each frame.

2. Industrial Preprocessing (Consistency)
It applies ROI cropping + greyscale preprocessing + dynamic horizontal masking so inference focuses on the product area with stable visual characteristics.

3. Operator Visualization (Human Readability)
It draws boxes/masks, zone-wise Good/Bad labels, confidence tables, FPS/status text, and hover tooltips.

4. Production Logging & Modes (Operational Use)
It supports live/image/video modes and stores inspection data to SQLite for traceability.
**

> **Script**: `scripts/defect_detector.py`  
> **Purpose**: Run defect detection inference in live/image/video modes and generate production-ready visual + logged outputs.  
> **When to use**: During validation and deployment/inference stages.

---

## Table of Contents

- [Overview](#overview)
- [Imports & Dependencies](#imports--dependencies)
- [DefectDetector Class](#defectdetector-class)
  - [Initialization](#initialization--__init__)
  - [Loading Config](#loading-config--_load_config)
  - [Warmup](#warmup--_warmup)
  - [Detection Pipeline](#detection-pipeline--detect)
  - [Image File Inference](#image-file-inference--detect_from_file)
  - [Video Inference](#video-inference--detect_from_video)
  - [Drawing Results](#drawing-results--draw_results)
- [InspectionLogger Class](#inspectionlogger-class)
- [ProductionInspectionSystem Class](#productioninspectionsystem-class)
- [CLI Entry Point (`main`)](#cli-entry-point-main)
- [How to Run](#how-to-run)
- [How It Connects to Other Scripts](#how-it-connects-to-other-scripts)

---

## Overview

This script has **4 main components**:

```
┌──────────────────────────────────────────────────────────┐
│                   defect_detector.py                     │
│                                                          │
│  ┌──────────────┐   ┌────────────────────┐              │
│  │ DefectDetector│   │ InspectionLogger   │              │
│  │ (YOLO infer)  │   │ (SQLite logging)   │              │
│  └──────┬───────┘   └─────────┬──────────┘              │
│         │                      │                         │
│         └──────────┬───────────┘                         │
│                    ▼                                     │
│         ProductionInspectionSystem                       │
│         (camera + infer + display loop)                 │
│                    │                                     │
│                    ▼                                     │
│                  main()                                  │
│            (live / image / video)                        │
└──────────────────────────────────────────────────────────┘
```

1. **`DefectDetector`** — preprocessing + model inference + parsing + annotation
2. **`InspectionLogger`** — saves inspection records into SQLite
3. **`ProductionInspectionSystem`** — wraps camera + detector + logger for live use
4. **`main()`** — command line mode router (`live`, `image`, `video`)

---

## Imports & Dependencies

```python
import os                               # File/path checks
import json                             # Load config and serialize defects
import time                             # Measure inference time and FPS
import sqlite3                          # Local DB logging
from pathlib import Path                # Safe path construction
from datetime import datetime           # Timestamps
from typing import Dict, List, Optional, Tuple

import cv2                              # OpenCV: image/video IO + drawing
import numpy as np                      # Numeric arrays and geometry helpers
```

### Ultralytics import guard (graceful handling)

```python
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
```

> **Why this pattern?**  
> It prevents immediate crash at import time and allows explicit error handling in class initialization.

```python
from camera_capture import (
    get_camera, BaslerCamera, MockCamera, convert_to_greyscale
)
```

> `convert_to_greyscale` is reused so inference preprocessing stays consistent with your pipeline assumptions.

---

## DefectDetector Class

### Initialization — `__init__`

```python
class DefectDetector:
    def __init__(self, model_path, config_path=None,
                 conf_threshold=0.03, iou_threshold=0.2, device='0'):
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics not installed")

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.config = self._load_config(config_path)

        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.is_seg = getattr(self.model, 'task', 'detect') == 'segment'
        self._warmup()
```

> **Key idea:** constructor does all critical setup once (config, model load, task type detection, warmup), so runtime loop remains fast.

---

### Loading Config — `_load_config`

```python
def _load_config(self, config_path: str) -> dict:
    default_config = {
        "model": {"confidence_threshold": 0.05, "iou_threshold": 0.20, "image_size": 640},
        "inspection": {"save_images": True, "save_path": "logs/inspections",
                       "log_to_database": True, "database_path": "logs/inspections.db"}
    }
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return default_config
```

> **Pattern used:** safe fallback defaults if external config is missing.

---

### Warmup — `_warmup`

```python
def _warmup(self, iterations: int = 3):
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(iterations):
        self.model(dummy, verbose=False)
```

> **Why warmup matters:** first forward pass often has overhead (kernel initialization, memory allocation). Warmup reduces first-frame latency spike.

---

### Detection Pipeline — `detect`

This is the core of the file.

#### Step 1: ROI crop

- Reads `inspection.roi` as `[ymin, ymax, xmin, xmax]`
- Clamps ROI to image bounds
- Crops inference region
- Stores offsets to map detections back to original frame coordinates

#### Step 2: Greyscale preprocessing

```python
grey_image = convert_to_greyscale(crop_image)
```

#### Step 3: Dynamic horizontal focus mask

- Optional `dynamic_margin` removes top/bottom noise
- Uses blur + Canny + morphology + largest contour
- Builds a central horizontal polygon (`core_box`)
- Masks image so YOLO sees focused region only

#### Step 4: Model inference

```python
results = self.model(
    masked_image,
    conf=self.conf_threshold,
    iou=self.iou_threshold,
    device=self.device,
    verbose=False
)
```

#### Step 5: Parse outputs

- Iterates YOLO boxes
- Converts class ID to class name
- Restores bbox coordinates to full image space using ROI offsets
- Builds normalized bbox
- If segmentation model: extracts and stores mask polygon
- Sorts defects by confidence descending

#### Step 6: Return structured result

Returned dict includes:
- `is_defective`
- `defect_count`
- `defects` (list of class/conf/bbox/mask data)
- `inference_time_ms`
- `timestamp`
- `dynamic_core_box`

> **Important:** the script supports both detection and segmentation models transparently.

---

### Image File Inference — `detect_from_file`

```python
image = cv2.imread(image_path)
if image is None:
    return {'error': f'Failed to load image: {image_path}'}
result = self.detect(image)
result['image_path'] = image_path
```

Simple wrapper around `detect()` with file loading and error safety.

---

### Video Inference — `detect_from_video`

- Opens input video
- Optional output writer (`mp4v`)
- Frame loop: read -> detect -> draw -> display
- Writes output if enabled
- Quit early with `q`
- Always releases capture/writer and closes windows in `finally`

---

### Drawing Results — `draw_results`

This method converts raw result dict into operator-friendly overlays.

#### What it draws:

1. **ROI boundary** (if configured)
2. **Dynamic core polygon**
3. **Defect visualization**
   - segmentation: semi-transparent filled mask + edge
   - detection: bbox rectangle
4. **Hover tooltip**
   - when mouse is over defect region, shows class + confidence
5. **Zone analysis**
   - splits part into 4 vertical zones
   - marks each zone as Good/Bad
   - draws per-zone mini table with top 2 defects
6. **Global status**
   - `PASS` / `REJECT`
   - inference time text

> This function is long because it packs visualization logic, interaction (hover), and zone analytics into one place.

---

## InspectionLogger Class

`InspectionLogger` is responsible for persistent inspection records.

### `_init_db`

- Creates `inspections` table if it does not exist
- Stores timestamp, image path, result, defect count/details, confidence, latency, shift/batch metadata

### `log`

- Computes max confidence from detected defects
- Inserts one row per inspection result

### `get_stats`

- Aggregates PASS/REJECT counts for last N hours
- Returns totals + defect rate

> This gives lightweight production traceability without external DB infrastructure.

---

## ProductionInspectionSystem Class

This class wires everything together for factory usage.

### Constructor

- Creates `DefectDetector`
- Gets camera using `get_camera(...)` (real or mock)
- Creates `InspectionLogger`

### `inspect_once`

- Captures one frame
- Runs detection
- Optionally saves annotated image under `logs/inspections/{pass|reject}`
- Logs result to SQLite

### `start` (live loop)

- Connects camera, starts streaming
- Repeats:
  - capture frame
  - resize to `1280x1280`
  - detect defects
  - draw overlays
  - show FPS + defect count
  - allow keyboard controls:
    - `q` quit
    - `v` toggle display (colour/greyscale)
    - `s` save snapshot
- Cleans up resources on exit

### `stop`

- Stops loop
- Disconnects camera
- Prints last 24-hour session stats from DB

---

## CLI Entry Point (`main`)

The parser defines:
- `--mode`: `live`, `image`, `video`
- `--model`: path to `.pt`
- `--source`: image/video path (when needed)
- `--config`: config JSON path
- `--conf`: confidence threshold
- `--mock`: force mock camera

Mode behavior:
- **video**: requires `--source`, processes file, writes `*_output.mp4`
- **image**:
  - with `--source`: run file inference + hover-capable window
  - without `--source`: single camera capture mode + optional save
- **live**: starts continuous inspection system

```python
if __name__ == "__main__":
    main()
```

Standard Python entrypoint guard.

---

## How to Run

```bash
# Live inspection (camera feed)
python scripts/defect_detector.py --mode live --model runs/detect/.../best.pt

# Single image from file
python scripts/defect_detector.py --mode image --model runs/detect/.../best.pt --source path/to/image.jpg

# Single image from camera
python scripts/defect_detector.py --mode image --model runs/detect/.../best.pt

# Video processing
python scripts/defect_detector.py --mode video --model runs/detect/.../best.pt --source path/to/video.mp4

# With explicit config/conf
python scripts/defect_detector.py --mode live --model ... --config config/system_config.json --conf 0.03
```

---

## How It Connects to Other Scripts

```
camera_capture.py
       │
       │ get_camera(), convert_to_greyscale()
       ▼
defect_detector.py
       │
       ├── uses trained YOLO weights from training runs
       ├── writes outputs to logs/inspections/
       └── logs structured results into SQLite
```

In pipeline terms:

1. `camera_capture.py` -> acquires frames
2. `defect_detector.py` -> predicts and decides
3. logs/images -> operational QA evidence and analysis

---

*This guide follows the same template style as `Camera_capture.md`, adapted for `defect_detector.py`.*


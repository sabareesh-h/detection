# File Flow: camera_capture.py → defect_detector.py

> All paths are relative to `Image_detection/` (project root).

---

## Pipeline at a Glance

```
[Basler Camera]
      │
      ▼  camera_capture.py
scripts/dataset/raw/<label>/<label>_TIMESTAMP.png
      │
      │  (Annotate in CVAT → export YOLO .txt labels)
      ▼
scripts/dataset/labels/train/*.txt
scripts/dataset/images/train/*.png   (if CVAT also exports images)
      │
      ▼  prepare_dataset.py  (merge + split 70/15/15)
scripts/dataset/
├── raw/images/    ← merged master copy
├── raw/labels/    ← merged master labels
├── images/train|val|test/
└── labels/train|val|test/
      │
      ▼  augment_dataset.py  (optional, in-place)
scripts/dataset/images/train/<stem>_aug000.png  (+matching .txt)
      │
      ▼  train_model.py
runs/<experiment>/weights/best.pt
runs/<experiment>/weights/last.pt
      │
      ▼  defect_detector.py
scripts/logs/inspections/pass|reject/<status>_TIMESTAMP.png
scripts/logs/inspections.db
```

---

## Stage 1 — Capture images (`camera_capture.py`)

**Run:** `python scripts/camera_capture.py`

| Item | Detail |
|------|--------|
| Camera used | Basler (or MockCamera if unavailable) |
| Save location | `scripts/dataset/raw/<label>/` |
| Filename format | `<label>_YYYYMMDD_HHMMSS_ffffff.png` |
| Save modes | **original** (colour) or **greyscale** (CLAHE + rust darkening) |

**Example output:**
```
scripts/dataset/raw/good/good_20260324_112500_123456.png
scripts/dataset/raw/rust/rust_20260324_112600_789012.png
```

---

## Stage 2 — Annotate in CVAT (manual step)

1. Open CVAT and load your captured images.
2. Draw bounding boxes with the correct class labels.
3. Export as **YOLO 1.1** format.
4. Unzip the export and place files:
   - Label files (`.txt`) → `scripts/dataset/labels/train/`
   - Image files (`.png`) → `scripts/dataset/images/train/`

Each `.txt` label file contains one line per bounding box:
```
<class_id> <x_center> <y_center> <width> <height>
```
*(all values normalised 0–1)*

---

## Stage 3 — Merge & Split (`prepare_dataset.py`)

**Run:** `python scripts/prepare_dataset.py`

Reads from `scripts/dataset/labels/train/` + `scripts/dataset/images/train/`, pairs each image with its label by filename stem, then:

| Output folder | Content | Split ratio |
|---------------|---------|-------------|
| `scripts/dataset/raw/images/` | All merged images (master copy) | 100% |
| `scripts/dataset/raw/labels/` | All merged labels (master copy) | 100% |
| `scripts/dataset/images/train/` | Training images | 70% |
| `scripts/dataset/images/val/` | Validation images | 15% |
| `scripts/dataset/images/test/` | Test images | 15% |
| `scripts/dataset/labels/train|val|test/` | Matching labels | same split |

> Files are **copied** (not moved), so originals remain in `raw/`.

---

## Stage 4 — Augment (`augment_dataset.py`) *(optional)*

**Run:** `python scripts/augment_dataset.py --input scripts/dataset/images/train --labels scripts/dataset/labels/train --multiplier 2`

For each original image, generates N augmented copies (rotation ±15°, CLAHE) **in the same folder**:

```
scripts/dataset/images/train/good_20260324_112500_123456_aug000.png
scripts/dataset/images/train/good_20260324_112500_123456_aug001.png
scripts/dataset/labels/train/good_20260324_112500_123456_aug000.txt
scripts/dataset/labels/train/good_20260324_112500_123456_aug001.txt
```

---

## Stage 5 — Train (`train_model.py`)

**Run:** `python scripts/train_model.py`

Reads dataset from `config/dataset.yaml` (which now points to `scripts/dataset/`).

| Output | Location |
|--------|----------|
| Best model weights | `runs/<experiment>/weights/best.pt` |
| Last epoch weights | `runs/<experiment>/weights/last.pt` |
| Training plots | `runs/<experiment>/` |

---

## Stage 6 — Detect (`defect_detector.py`)

**Run:**
```
# Live camera feed
python scripts/defect_detector.py --mode live --model runs/<experiment>/weights/best.pt

# Single image from camera
python scripts/defect_detector.py --mode image --model runs/<experiment>/weights/best.pt

# Process a video file
python scripts/defect_detector.py --mode video --model runs/<experiment>/weights/best.pt --source video.mp4
```

| Action | File saved to |
|--------|--------------|
| Press `s` (snapshot in live mode) | `scripts/logs/inspections/pass/` or `reject/` |
| Image mode (press `s`) | `scripts/logs/capture_TIMESTAMP.png` |
| Video mode output | `<input_name>_output.mp4` (same folder as input) |
| Every result logged | `scripts/logs/inspections.db` (SQLite) |

---

## Quick Reference: Where to Put New Data

| Scenario | Drop files here |
|----------|----------------|
| New images captured from camera | Auto-saved to `scripts/dataset/raw/<label>/` |
| New annotated images from CVAT | `.png` → `scripts/dataset/images/train/`  `.txt` → `scripts/dataset/labels/train/` |
| Run `prepare_dataset.py` after adding new data | Automatically re-splits everything |

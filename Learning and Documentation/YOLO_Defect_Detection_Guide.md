# YOLO Defect Detection — Complete Guide
> Full conversation covering data gathering to production inference for metal surface defect detection (rust & dent)

---

## Table of Contents

1. [Full Pipeline Overview](#1-full-pipeline-overview)
2. [Stage 1 — Data Gathering](#2-stage-1--data-gathering)
3. [Stage 2 — Annotation](#3-stage-2--annotation)
4. [Stage 3 — Pre-processing & Augmentation](#4-stage-3--pre-processing--augmentation)
5. [Stage 4 — Model Selection & Configuration](#5-stage-4--model-selection--configuration)
6. [Stage 5 — Training & Fine-tuning](#6-stage-5--training--fine-tuning)
7. [Stage 6 — Evaluation](#7-stage-6--evaluation)
8. [Stage 7 — Optimization & Deployment](#8-stage-7--optimization--deployment)
9. [Accuracy Improvement Techniques](#9-accuracy-improvement-techniques)
10. [Hyperparameter Search — W&B & Ray Tune](#10-hyperparameter-search--wb--ray-tune)
11. [MVTec Dataset — What It Is and How to Use It](#11-mvtec-dataset--what-it-is-and-how-to-use-it)
12. [Good vs Bad Parts — Correct Class Strategy](#12-good-vs-bad-parts--correct-class-strategy)
13. [Bounding Box Strategy — Tight vs Loose](#13-bounding-box-strategy--tight-vs-loose)
14. [Class Imbalance Handling](#14-class-imbalance-handling)
15. [Weights & Biases (W&B) — Full Implementation](#15-weights--biases-wb--full-implementation)
16. [Per-Size Analysis & P2 Detection Head](#16-per-size-analysis--p2-detection-head)
17. [Tiling Inference with SAHI](#17-tiling-inference-with-sahi)

---

## 1. Full Pipeline Overview

```
Data Gathering → Annotation → Pre-processing & Augmentation
→ Model Selection → Training & Fine-tuning
→ Evaluation → Optimization & Deployment
↑___________________________(continuous improvement loop)
```

---

## 2. Stage 1 — Data Gathering

This is where most projects succeed or fail. Companies pull data from three sources:

### Real Production Images
Captured directly on the manufacturing line using machine vision cameras. A common starting target is **500–2,000 images per defect class**. Collect from multiple shifts, lighting conditions, and machine states to prevent the model from learning spurious correlations.

### Synthetic Data
Use tools like NVIDIA Omniverse, Blender, or domain-specific simulators to render photorealistic defects on 3D part models. This fills class imbalance gaps — critical defects like cracks are rare in real production.

### Edge Case Mining
After an initial model is deployed, run all production images through it and flag low-confidence predictions or false positives. These **hard examples** are added back to training — called **hard negative mining**.

---

## 3. Stage 2 — Annotation

YOLO label format (normalized):
```
class x_center y_center width height
```
All values between 0 and 1.

### Tools
- **Label Studio** — open source, web-based
- **Roboflow** — cloud-based with augmentation
- **CVAT** — open source, feature-rich

### Key Practices
- Use multi-annotator consensus (2–3 annotators) for ambiguous defects
- Measure inter-annotator agreement via IoU
- Define a clear annotation ontology — what exactly counts as "scratch" vs "mark"
- Use active learning loops where the model pre-labels images and humans only correct

---

## 4. Stage 3 — Pre-processing & Augmentation

### Geometric Augmentations
- Flip, rotate, shear
- Crop, perspective warp
- Scale jitter

### Photometric Augmentations
- Brightness, contrast
- Gaussian blur, noise
- CLAHE, gamma correction

### Advanced Augmentations
- **Mosaic** — stitches 4 images into one training sample
- **MixUp** — blends two images together
- **CutOut** — randomly masks patches
- **Copy-paste** — pastes annotated defects onto clean images

### Industry-Specific Trick: Copy-Paste Augmentation
Take an annotated defect (e.g., a scratch on metal), cut it out, and paste it onto clean part images at different positions and orientations. Artificially balances classes without requiring new real-world defects.

> **Warning:** Do NOT over-augment — excessive distortion makes defects unrecognisable to both annotators and the model.

---

## 5. Stage 4 — Model Selection & Configuration

### Which YOLO Version?

| Model | Use Case |
|---|---|
| YOLOv8n / v8s | Edge devices (Jetson Nano, Raspberry Pi) |
| YOLOv8m | Balanced — most factory deployments |
| YOLOv8l / v8x | High-accuracy QC stations with GPU server |
| YOLOv8-seg | Irregular defects needing pixel-level masks |

### Architecture Tweaks

**For small defect detection:** Add an extra P2 detection head (stride 4) to the neck (PAN-FPN). This is the most common architectural change for industrial use.

**Custom backbone:** Some teams replace the backbone with one pretrained on industrial textures (e.g., trained on MVTec dataset) rather than ImageNet-pretrained weights.

**Anchor configuration (older YOLO):** Run `autoanchor` on your actual dataset — default COCO anchors are sized for natural images, not small surface defects.

---

## 6. Stage 5 — Training & Fine-tuning

### Transfer Learning Strategy
1. Start from COCO-pretrained weights
2. Freeze the backbone for the first 10–20 epochs
3. Unfreeze and fine-tune the full network

### Two-Phase Training
```python
from ultralytics import YOLO

# Phase 1 — freeze backbone, train head only
model = YOLO("yolov8m.pt")
model.train(
    data="metal_defects.yaml",
    epochs=20,
    imgsz=1280,
    lr0=1e-3,
    freeze=10,
    batch=16,
    project="metal_detection",
    name="phase1_frozen"
)

# Phase 2 — unfreeze all, fine-tune with low LR
model2 = YOLO("metal_detection/phase1_frozen/weights/best.pt")
model2.train(
    data="metal_defects.yaml",
    epochs=80,
    imgsz=1280,
    lr0=1e-4,
    freeze=0,
    batch=8,
    project="metal_detection",
    name="phase2_full_finetune"
)
```

### Key Hyperparameters to Tune

| Parameter | What it does | Typical range |
|---|---|---|
| `lr0` | Initial learning rate | 1e-4 to 1e-2 |
| `weight_decay` | Regularisation, prevents overfitting | 1e-5 to 1e-3 |
| `mosaic` | Mosaic augmentation probability | 0.0 to 1.0 |
| `warmup_epochs` | Ramp-up period | 1 to 10 |
| `imgsz` | Input resolution | 640 to 1280 |
| `cls_pw` | Class positive weight (imbalance) | 1.0 to 10.0 |

### Class Imbalance — Weighted Focal Loss
```python
model.train(
    data="metal_defects.yaml",
    epochs=100,
    imgsz=1280,
    cls_pw=8.2,       # weight for rare class (dent)
    copy_paste=0.5,   # augment rare class synthetically
    mosaic=1.0,
)
```

---

## 7. Stage 6 — Evaluation

### Key Metrics

- **mAP@0.5** — headline metric
- **Recall at fixed precision** — in QC, missing a defect (false negative) costs more than a false alarm — operate at threshold that achieves ~99% recall
- **Confusion matrix per class** — reveals which defect types are confused with each other
- **Per-size analysis** — split evaluation into small/medium/large defect bins

```python
from ultralytics import YOLO

model = YOLO("metal-inspection/best.pt")
metrics = model.val(data="metal_defects.yaml")

print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'mAP50':>10}")
for i, name in enumerate(["rust", "dent"]):
    p  = metrics.box.p[i]
    r  = metrics.box.r[i]
    ap = metrics.box.ap50[i]
    print(f"{name:<10} {p:>10.3f} {r:>10.3f} {ap:>10.3f}")
```

---

## 8. Stage 7 — Optimization & Deployment

### Export Formats

| Format | Hardware | Speedup |
|---|---|---|
| TensorRT INT8 | NVIDIA Jetson / GPU server | 3–8× faster than FP32 |
| ONNX Runtime | Cross-platform CPU/GPU | Easy integration |
| OpenVINO | Intel CPUs / Movidius VPU | Low-power edge |

### TensorRT Export
```python
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="engine", int8=True, data="metal_defects.yaml")
# Produces: best.engine — ready for Jetson/GPU server
```

### Deployment Modes
- **Inline on conveyor** — <50ms latency required, TensorRT on Jetson
- **REST API server** — MES/ERP integration
- **Smart camera** — Cognex / Basler Dart embedded

---

## 9. Accuracy Improvement Techniques

| Technique | Impact | Notes |
|---|---|---|
| Tight bounding boxes (not whole surface) | Very high | Most important labeling rule |
| 500+ images per defect class | Very high | Data volume matters most |
| `imgsz=1280` instead of 640 | High | Rust spots are small |
| `freeze=10` during phase 1 | Medium | Protects texture features |
| Multi-scale training | Medium | Random `imgsz` each batch |
| Test-time augmentation (TTA) | +1–3% mAP | 3× slower inference |
| Ensemble inference (WBF) | +2–4% mAP | 2× slower inference |
| MVTec pre-training | Low–Medium | Only after basics are solid |
| P2 detection head | Medium | Only if >10% tiny defects |
| SAHI tiling | High | Only for high-res cameras (2MP+) |

---

## 10. Hyperparameter Search — W&B & Ray Tune

### What They Are

**Weights & Biases (W&B)** — tracks every training run visually, shows live charts (loss, mAP per epoch), and runs automated sweeps to find best hyperparameters. Best for: visibility + experiment tracking.

**Ray Tune** — runs many trials in parallel across multiple GPUs, stops bad trials early. Best for: large-scale parallel search. Limited value on a single GPU.

### Cost
- **W&B** — free forever for personal use (unlimited runs, 100GB storage)
- **Ray Tune** — fully open source, no limitations

### W&B Setup
```bash
pip install wandb ultralytics
```

```python
import wandb
wandb.login()  # sign up free at wandb.ai
```

### Simple Training with W&B Tracking
```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
model.train(
    data="metal_defects.yaml",
    epochs=100,
    imgsz=1280,
    batch=8,
    project="metal-inspection",   # W&B picks this up automatically
    name="baseline-run"
)
```

### W&B Hyperparameter Sweep
```python
import wandb
from ultralytics import YOLO

sweep_config = {
    "method": "bayes",
    "metric": {"name": "metrics/mAP50(B)", "goal": "maximize"},
    "parameters": {
        "lr0":        {"distribution": "log_uniform_values", "min": 0.0001, "max": 0.01},
        "cls_pw":     {"values": [1.0, 4.0, 8.0, 12.0]},
        "freeze":     {"values": [0, 5, 10, 15]},
        "imgsz":      {"values": [640, 1280]},
        "copy_paste": {"distribution": "uniform", "min": 0.0, "max": 0.5}
    }
}

def run_trial():
    run  = wandb.init()
    cfg  = run.config

    model   = YOLO("yolov8m.pt")
    results = model.train(
        data="metal_defects.yaml",
        epochs=30,
        imgsz=cfg.imgsz,
        batch=8,
        freeze=cfg.freeze,
        lr0=cfg.lr0,
        cls_pw=cfg.cls_pw,
        copy_paste=cfg.copy_paste,
        project="metal-inspection",
        name=run.name,
    )

    wandb.log({"final_mAP50": results.results_dict["metrics/mAP50(B)"]})
    wandb.finish()

sweep_id = wandb.sweep(sweep_config, project="metal-inspection")
wandb.agent(sweep_id, function=run_trial, count=20)
```

### Final Training with Best Config
```python
best_config = {
    "lr0":        0.005,
    "cls_pw":     8.0,
    "freeze":     10,
    "imgsz":      1280,
    "copy_paste": 0.3
}

model = YOLO("yolov8m.pt")
model.train(
    data="metal_defects.yaml",
    epochs=150,
    **best_config,
    project="metal-inspection",
    name="final-best-config"
)
```

---

## 11. MVTec Dataset — What It Is and How to Use It

### Key Concept

**MVTec has no bounding boxes — and that's fine.** It is used only to pre-train the backbone (feature extractor), not the detection head.

| | COCO Pretrain | MVTec Pretrain |
|---|---|---|
| What it teaches | General vision: shapes, edges, objects | Industrial textures: metal, wood, leather, tile |
| Has bounding boxes | Yes | No |
| Used for | Full YOLO training | Backbone feature learning only |

> **COCO absolutely HAS bounding boxes.** When you load `yolov8m.pt`, those weights came from training on COCO's bounding boxes.

### The Analogy
- **COCO backbone** = someone who spent 5 years looking at cats, dogs, and cars — never seen metal
- **MVTec backbone** = someone who spent 5 years examining metal sheets, wood panels, leather — knows industrial surfaces deeply

### When to Use MVTec
Only add MVTec pre-training if your accuracy is still poor after:
- Proper tight bounding box annotation
- 500+ images per class
- `imgsz=1280` training
- `cls_pw` imbalance fix

### Download MVTec
```python
import kagglehub
path = kagglehub.dataset_download("ipythonx/mvtec-ad")
```

Or manually:
```bash
wget https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_anomaly_detection.tar.xz
tar -xf mvtec_anomaly_detection.tar.xz
```

### Loading MVTec Weights into YOLO with Layer Freezing
```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt")

# Freeze first 10 layers (edge/texture detectors)
for i, (name, param) in enumerate(model.model.named_parameters()):
    layer_num = int(name.split('.')[1]) if name.split('.')[1].isdigit() else 999
    if layer_num < 10:
        param.requires_grad = False

frozen   = sum(1 for p in model.model.parameters() if not p.requires_grad)
trainable = sum(1 for p in model.model.parameters() if p.requires_grad)
print(f"Frozen: {frozen} | Trainable: {trainable}")
```

### What Each Layer Group Learns

```
Layer 0–3   →  Edges, gradients       (always freeze)
Layer 4–7   →  Textures, patterns     (freeze for industrial)
Layer 8–15  →  Part shapes            (fine-tune)
Layer 16+   →  Task-specific          (always train)
```

---

## 12. Good vs Bad Parts — Correct Class Strategy

### The Core Rule
**Never create a "good" class.** YOLO detects objects — "good" means nothing is there to find.

```
YOLO finds rust or dent box  →  Part is DEFECTIVE — reject
YOLO finds nothing           →  Part is GOOD — pass
```

### Your dataset.yaml
```yaml
path: ./metal_dataset
train: images/train
val:   images/val
test:  images/test

nc: 2                    # NOT 3 — no "good" class
names: ["rust", "dent"]
```

### Label File Examples

**Good part** → `good_part_001.txt`
```
(empty file — no lines)
```

**Rust defect** → `rust_part_001.txt`
```
0 0.52 0.44 0.18 0.12
│  │    │    │    └─ box height (normalized)
│  │    │    └────── box width (normalized)
│  │    └─────────── y center (normalized)
│  └──────────────── x center (normalized)
└─────────────────── class id: 0 = rust
```

**Both defects** → `both_defects_001.txt`
```
0 0.38 0.35 0.14 0.10
1 0.72 0.68 0.16 0.09
```

---

## 13. Bounding Box Strategy — Tight vs Loose

### The Rule
**Always box only the specific defect area. Never the entire surface.**

Adding 5–10px padding around the visible defect edge is ideal:
- **Too tight (0px)** — clips defect edges, model misses boundary features
- **Correct (~8px padding)** — full defect inside box, tiny clean border
- **Too loose (50px+)** — includes clean surface, confuses the model

### Why Tight Boxes Work Better
When YOLO trains, it crops the region inside your bounding box and asks "what makes this region look like rust?" If your box covers the entire surface, clean metal pixels inside the box act as **noise that directly fights the learning signal**.

### Random Placement is Good
Defects appearing randomly across training images forces YOLO to learn **what rust/dent looks like** rather than **where defects usually appear**. This prevents positional bias.

### Bounding Box vs Segmentation

| Defect Type | Recommendation |
|---|---|
| Dent (oval/round shape) | Bounding box is sufficient |
| Rust (irregular, spreading) | Consider YOLOv8-seg polygon mask |

> Start with bounding boxes for both. Switch rust to segmentation only if mAP is still poor after proper training.

---

## 14. Class Imbalance Handling

### Step 1 — Check Your Distribution
```python
import os
from pathlib import Path
from collections import defaultdict

labels_dir  = Path("metal_dataset/labels/train")
class_names = ["rust", "dent"]
class_counts = defaultdict(int)
total_boxes  = 0

for label_file in labels_dir.glob("*.txt"):
    with open(label_file) as f:
        for line in f:
            line = line.strip()
            if line:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1
                total_boxes += 1

for class_id, name in enumerate(class_names):
    count = class_counts[class_id]
    pct   = (count / total_boxes * 100) if total_boxes else 0
    print(f"{name}: {count} boxes  {pct:.1f}%")

ratio = max(class_counts.values()) / max(min(class_counts.values()), 1)
print(f"Imbalance ratio: {ratio:.1f}×")
```

### Decision Guide

| Ratio | Action |
|---|---|
| Under 3× | Balanced — no cls_pw needed |
| 3×–10× | Moderate — apply cls_pw |
| Over 10× | Severe — cls_pw + collect more data |

### Step 2 — Calculate cls_pw Weight
```python
rust_count = class_counts[0]   # e.g. 820
dent_count = class_counts[1]   # e.g. 100
total = rust_count + dent_count

weight_rust = total / (2 * rust_count)
weight_dent = total / (2 * dent_count)

min_w            = min(weight_rust, weight_dent)
weight_rust_norm = weight_rust / min_w   # → 1.0
weight_dent_norm = weight_dent / min_w   # → 8.2

print(f"rust weight: {weight_rust_norm:.2f}")
print(f"dent weight: {weight_dent_norm:.2f}")
```

### Step 3 — Apply in Training
```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
model.train(
    data="metal_defects.yaml",
    epochs=100,
    imgsz=1280,
    batch=8,
    freeze=10,
    cls_pw=8.2,        # dent/rust imbalance ratio
    copy_paste=0.5,    # synthetically boost rare class
    mosaic=1.0,
    project="metal-inspection",
    name="v1_balanced"
)
```

### Expected Result After Fixing Imbalance

```
Before cls_pw:
  rust  — precision: 0.91  recall: 0.88  mAP50: 0.90
  dent  — precision: 0.61  recall: 0.34  mAP50: 0.42  ← ignored

After cls_pw=8.2:
  rust  — precision: 0.88  recall: 0.86  mAP50: 0.87  ← tiny drop, acceptable
  dent  — precision: 0.79  recall: 0.71  mAP50: 0.76  ← massive improvement
```

---

## 15. Weights & Biases (W&B) — Full Implementation

See [Section 10](#10-hyperparameter-search--wb--ray-tune) for the complete W&B implementation.

### What W&B Tracks Automatically
- Loss curves (train & val) per epoch
- mAP per class per epoch
- Precision and recall curves
- Confusion matrix
- All hyperparameters used
- Model weights (optional)

### How Bayesian Search Works
```
Trial 1:  lr=0.01,  cls_pw=1.0  → mAP dent: 0.42  (bad)
Trial 2:  lr=0.005, cls_pw=4.0  → mAP dent: 0.61  (better, search near here)
Trial 3:  lr=0.005, cls_pw=8.0  → mAP dent: 0.74  (good)
Trial 4:  lr=0.004, cls_pw=8.2  → mAP dent: 0.77  (even better)
...
Trial 20: lr=0.005, cls_pw=8.2  → mAP dent: 0.79  ← W&B picks this as best
```

W&B learns from each trial which direction to search next — not random guessing.

---

## 16. Per-Size Analysis & P2 Detection Head

### Default YOLO Detection Scales

| Head | Stride | Detects |
|---|---|---|
| P3 | 8 | ~8–80px objects |
| P4 | 16 | ~80–200px objects |
| P5 | 32 | ~200px+ objects |
| **P2 (extra)** | **4** | **<8px objects (tiny defects)** |

### Step 1 — Analyse Your Defect Sizes
```python
import os
from pathlib import Path
import statistics
from PIL import Image

labels_dir  = Path("metal_dataset/labels/train")
images_dir  = Path("metal_dataset/images/train")
class_names = ["rust", "dent"]
sizes       = {"rust": [], "dent": []}

sample_imgs  = list(images_dir.glob("*.jpg"))
sample       = Image.open(sample_imgs[0])
img_w, img_h = sample.size

for label_file in labels_dir.glob("*.txt"):
    with open(label_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts       = line.split()
            class_id    = int(parts[0])
            px_w        = float(parts[3]) * img_w
            px_h        = float(parts[4]) * img_h
            defect_size = min(px_w, px_h)
            sizes[class_names[class_id]].append(defect_size)

for name, size_list in sizes.items():
    if not size_list:
        continue
    total  = len(size_list)
    tiny   = sum(1 for s in size_list if s < 32)
    print(f"{name.upper()}: {tiny/total*100:.1f}% tiny (<32px)")
    print(f"  Median: {statistics.median(size_list):.1f}px")
    print(f"  Smallest: {min(size_list):.1f}px")
```

### Decision Guide

| Tiny defects % | Action |
|---|---|
| Under 5% | Default 3 heads fine — skip P2 |
| 5–20% | Consider adding P2 |
| Over 20% | Definitely add P2 |

### Adding P2 Head (custom model YAML)

Create `yolov8m-p2.yaml`:
```yaml
nc: 2   # rust, dent

scales:
  m: [0.67, 0.75, 768]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]       # 0 P1/2
  - [-1, 1, Conv, [128, 3, 2]]      # 1 P2/4
  - [-1, 3, C2f, [128, True]]       # 2
  - [-1, 1, Conv, [256, 3, 2]]      # 3 P3/8
  - [-1, 6, C2f, [256, True]]       # 4
  - [-1, 1, Conv, [512, 3, 2]]      # 5 P4/16
  - [-1, 6, C2f, [512, True]]       # 6
  - [-1, 1, Conv, [512, 3, 2]]      # 7 P5/32
  - [-1, 3, C2f, [512, True]]       # 8
  - [-1, 1, SPPF, [512, 5]]         # 9

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]             # 12

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, C2f, [256]]             # 15 P3

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 3, C2f, [128]]             # 18 P2 ← NEW

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 15], 1, Concat, [1]]
  - [-1, 3, C2f, [256]]             # 21

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]             # 24

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]             # 27

  - [[18, 21, 24, 27], 1, Detect, [nc]]   # 4 heads
```

Train with it:
```python
from ultralytics import YOLO

model = YOLO("yolov8m-p2.yaml").load("yolov8m.pt")
model.train(
    data="metal_defects.yaml",
    epochs=100,
    imgsz=1280,
    batch=6,       # smaller batch — P2 uses more memory
    freeze=10,
    cls_pw=8.2,
    project="metal-inspection",
    name="p2-head-model"
)
```

### Recommended Order
```
1. Train at imgsz=1280 (default 3 heads)
2. Run size analysis script
3. If tiny defects < 10% → done
4. If tiny defects > 10% → add P2 head and retrain
5. Compare mAP before and after — keep whichever is higher
```

---

## 17. Tiling Inference with SAHI

### The Problem
When a 2560×1920 camera image is resized to 640px for YOLO:
- An 18px rust spot becomes a **4px dot**
- YOLO completely misses it
- Information is permanently destroyed by the resize

### How Tiling Solves It
1. Slice the full-resolution image into overlapping 640×640 tiles
2. Run YOLO on each tile at full resolution
3. Map all detections back to original image coordinates
4. Apply NMS to remove duplicates from overlapping tiles

**Why 20% overlap?** A defect on a tile border gets cut in half and missed. Overlap ensures every defect falls fully inside at least one tile.

### When to Use Tiling

| Camera Resolution | Action |
|---|---|
| 640×640 or lower | Not needed |
| 1280×1280 | Try `imgsz=1280` first |
| 2MP+ (1920×1080+) | Yes — tiling will noticeably help |
| 4K+ industrial camera | Definitely use tiling |

### Installation
```bash
pip install sahi ultralytics
```

### Approach 1 — SAHI (Easiest)
```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="metal-inspection/final-best-config/weights/best.pt",
    confidence_threshold=0.3,
    device="cuda",
)

result = get_sliced_prediction(
    "test_metal_part.jpg",
    detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

print(f"Total defects found: {len(result.object_prediction_list)}")
for pred in result.object_prediction_list:
    print(f"  {pred.category.name}: conf={pred.score.value:.2f}  bbox={pred.bbox.to_xyxy()}")

result.export_visuals(export_dir="results/", file_name="detected_part")
```

### Approach 2 — Manual Tiling (Full Control)
```python
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision.ops import nms

def run_tiled_inference(
    image_path,
    model_path,
    tile_size=640,
    overlap=0.2,
    conf_threshold=0.3,
    iou_threshold=0.45
):
    model  = YOLO(model_path)
    image  = cv2.imread(image_path)
    img_h, img_w = image.shape[:2]
    stride = int(tile_size * (1 - overlap))
    all_boxes = []

    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            x2   = min(x + tile_size, img_w)
            y2   = min(y + tile_size, img_h)
            x1   = max(x2 - tile_size, 0)
            y1   = max(y2 - tile_size, 0)
            tile = image[y1:y2, x1:x2]

            results = model.predict(tile, conf=conf_threshold, verbose=False)

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    tx1, ty1, tx2, ty2 = box.xyxy[0].tolist()
                    all_boxes.append({
                        "x1": tx1 + x1, "y1": ty1 + y1,
                        "x2": tx2 + x1, "y2": ty2 + y1,
                        "conf":      float(box.conf[0]),
                        "class_id":  int(box.cls[0]),
                        "class_name": model.names[int(box.cls[0])]
                    })

    if not all_boxes:
        return []

    boxes_t  = torch.tensor([[b["x1"],b["y1"],b["x2"],b["y2"]] for b in all_boxes])
    scores_t = torch.tensor([b["conf"] for b in all_boxes])
    keep     = nms(boxes_t, scores_t, iou_threshold)

    return [all_boxes[i] for i in keep.tolist()]


def draw_and_save(image_path, detections, output_path):
    image  = cv2.imread(image_path)
    colors = {"rust": (20, 150, 186), "dent": (186, 80, 20)}

    for det in detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        color = colors.get(det["class_name"], (128,128,128))
        cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)
        cv2.putText(image, f"{det['class_name']} {det['conf']:.2f}",
                    (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    verdict       = "DEFECTIVE" if detections else "GOOD"
    verdict_color = (0,0,200) if detections else (0,180,0)
    cv2.putText(image, verdict, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, verdict_color, 3)
    cv2.imwrite(output_path, image)


# Run
detections = run_tiled_inference(
    image_path="test_metal_part.jpg",
    model_path="metal-inspection/final-best-config/weights/best.pt",
)
draw_and_save("test_metal_part.jpg", detections, "results/result_tiled.jpg")

if detections:
    print(f"Part REJECTED — {len(detections)} defect(s) found")
    for d in detections:
        print(f"  {d['class_name']} conf={d['conf']:.2f}")
else:
    print("Part PASSED — no defects detected")
```

### Approach 3 — Batch Processing for Production
```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os, time

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="metal-inspection/final-best-config/weights/best.pt",
    confidence_threshold=0.3,
    device="cuda",
)

def inspect_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    images = [f for f in os.listdir(input_folder) if f.endswith((".jpg",".png",".bmp"))]

    passed = rejected = 0
    for fname in images:
        fpath  = os.path.join(input_folder, fname)
        result = get_sliced_prediction(
            fpath, detection_model,
            slice_height=640, slice_width=640,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        )
        defects = result.object_prediction_list
        if defects:
            rejected += 1
            result.export_visuals(export_dir=output_folder, file_name=fname.split(".")[0])
        else:
            passed += 1

    print(f"Passed : {passed}")
    print(f"Rejected: {rejected}")
    print(f"Defect rate: {rejected/(passed+rejected)*100:.1f}%")

inspect_folder("production_images/", "flagged_defects/")
```

---

## Summary — Recommended Order for Your Metal Project

```
Step 1  — Collect 500+ images per class (rust, dent)
Step 2  — Annotate with tight bounding boxes (8px padding)
          Use only 2 classes: rust=0, dent=1
          Leave good part label files empty
Step 3  — Run class distribution analysis
          Calculate cls_pw for imbalance
Step 4  — Train Phase 1: freeze=10, epochs=20, lr=1e-3
Step 5  — Train Phase 2: freeze=0,  epochs=80, lr=1e-4
Step 6  — Run W&B sweep (20 trials) to find best hyperparameters
Step 7  — Run size analysis — add P2 head only if >10% tiny defects
Step 8  — Check camera resolution — add SAHI tiling if 2MP+
Step 9  — Evaluate per-class mAP (rust and dent separately)
Step 10 — Export to TensorRT for production deployment
```

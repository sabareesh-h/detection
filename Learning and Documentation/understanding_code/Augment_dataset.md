# 🧪 Dataset Augmentation — Complete Code Walkthrough

**In an industrial defect-detection project, `augment_dataset.py` is the “Data Multiplier”.**
It takes your *real* labeled images (YOLO bounding boxes) and generates many realistic variations so YOLO learns to handle lighting changes, small rotations, blur/noise, and shadows that happen on the factory floor.

Here are the 4 main reasons why we need this file:

1. **More Training Data (Without More Labeling)**

Labeling defects is slow and expensive. This script multiplies your dataset size (e.g., 5× or 10×) using realistic transformations, while keeping bounding boxes correct.

1. **Better Generalization (Robust Model)**

Real environments change: lighting shifts, camera angle changes slightly, motion blur happens, shadows appear. Augmentation teaches the model: “the defect is still the defect even if conditions change.”

1. **Bounding Boxes Stay Correct (YOLO-Safe Augmentation)**

Randomly editing images is easy; editing images *and updating bounding boxes correctly* is the hard part. This script uses Albumentations with YOLO bbox support so labels remain valid after transformations.

1. **Preview Mode (Sanity Check Before You Generate Thousands of Files)**

It can show you augmented examples in windows **without saving** anything, so you can quickly verify the transformations and bounding boxes look correct.

> **Script**: `scripts/augment_dataset.py`  
> **Purpose**: Offline augmentation of YOLO training images while preserving YOLO bounding boxes.  
> **When to use**: After you have labeled images (YOLO `.txt` files) and before training (`train_model.py`).

---

## Table of Contents

- [Overview](#overview)
- [Imports & Dependencies](#imports--dependencies)
- [YOLO Label I/O](#yolo-label-io)
  - [Reading Labels](#reading-labels--read_yolo_labels)
  - [Writing Labels](#writing-labels--write_yolo_labels)
- [Augmentation Pipelines](#augmentation-pipelines)
  - [Light / Medium / Heavy](#light--medium--heavy--get_augmentation_pipeline)
  - [Why `min_visibility` Matters](#why-min_visibility-matters)
- [Augmentation Engine](#augmentation-engine)
  - [Augment One Image](#augment-one-image--augment_single_image)
  - [Augment Whole Dataset](#augment-whole-dataset--augment_dataset)
  - [Output Naming Strategy](#output-naming-strategy)
- [Preview Mode (No Saving)](#preview-mode-no-saving)
  - [Drawing Boxes](#drawing-boxes--draw_bboxes)
  - [Preview Loop](#preview-loop--preview_augmentations)
- [CLI (Command Line Interface)](#cli-command-line-interface)
- [How to Run](#how-to-run)
- [How It Connects to Other Scripts](#how-it-connects-to-other-scripts)

---

## Overview

At a high level, the script looks like this:

```
┌───────────────────────────────────────────────────────────────┐
│                     augment_dataset.py                         │
│                                                               │
│  ┌───────────────┐      ┌────────────────────┐                │
│  │ YOLO labels    │      │ Albumentations      │                │
│  │ read/write     │      │ augmentation pipeline│               │
│  └───────┬───────┘      └─────────┬──────────┘               │
│          │                          │                          │
│          ▼                          ▼                          │
│   augment_single_image()  →  saves image + updated .txt labels │
│          │                                                     │
│          ▼                                                     │
│      augment_dataset() → loops through a folder of images       │
│                                                               │
│   preview_augmentations() → visualize only (no file output)     │
└───────────────────────────────────────────────────────────────┘
```

You give it:

- `dataset/images/train/` (images: `.jpg/.png/...`)
- `dataset/labels/train/` (YOLO label files: same stem name, `.txt`)

And it produces:

- Augmented images: `*_aug000.png`, `*_aug001.png`, ...
- Matching augmented labels: `*_aug000.txt`, `*_aug001.txt`, ...

---

## Imports & Dependencies

The script depends on:

- **OpenCV (`cv2`)**: reading/writing images, drawing bounding boxes, showing preview windows
- **Albumentations**: augmentation library that can transform images *and* bboxes together
- **NumPy + random**: reproducibility and minor utilities

### Optional dependency pattern (Albumentations)

Just like `camera_capture.py` uses a “graceful import” for optional dependencies, this script does:

```python
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Error: albumentations not installed. Run: pip install albumentations>=1.3.0")
```

> **💡 Why this matters**
>
> - The script can still be imported without Albumentations installed (useful for tooling).
> - But if you actually try to run augmentation, it will stop with a clear instruction.

---

## YOLO Label I/O

YOLO label format (per line):

```
class_id  x_center  y_center  width  height
```

All coordinates are **normalized** (0–1) relative to image width/height.

### Reading Labels — `read_yolo_labels`

Key behavior:

- If the label file is missing, it returns an empty list.
- Parses each non-empty line into `[class_id, x_center, y_center, width, height]`.

```python
def read_yolo_labels(label_path: str) -> List[List[float]]:
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path, 'r') as f:
        for line in f:
            ...
            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]
            labels.append([class_id] + bbox)
    return labels
```

> **💡 Important constraint**
>
> The code assumes each label line has at least 5 parts. If a label file is corrupted, you may see exceptions later during augmentation.

### Writing Labels — `write_yolo_labels`

Writes one line per bbox and formats floats to 6 decimals:

```python
def write_yolo_labels(label_path: str, labels: List[List[float]]):
    with open(label_path, 'w') as f:
        for label in labels:
            class_id = int(label[0])
            bbox = label[1:5]
            f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
```

> **💡 Why the 6-decimal formatting?**
>
> YOLO label files are plain text, but keeping consistent formatting avoids “label noise” and makes diffs stable.

---

## Augmentation Pipelines

### Light / Medium / Heavy — `get_augmentation_pipeline`

This function builds an Albumentations `Compose` pipeline.

Key idea: **augmentation intensity is controlled by `level`**:

- **light**: safe, small changes (flip, mild brightness/contrast, mild noise)
- **medium** (default): typical factory variation (flip, small rotation, moderate lighting, some blur/noise, CLAHE, shadows, small HSV shifts)
- **heavy**: stronger changes (bigger rotations, more blur/noise, scale, piecewise affine distortion)

Example structure (simplified):

```python
if level == "light":
    transforms = [HorizontalFlip, RandomBrightnessContrast, GaussNoise]
elif level == "heavy":
    transforms = [Flip, Rotate(30°), BrightnessContrast(0.3), OneOf(noise/blur), ...]
else:
    transforms = [Flip, Rotate(15°), BrightnessContrast(0.2), OneOf(noise/blur), ...]
```

Then it wraps them with bbox support:

```python
pipeline = A.Compose(
    transforms,
    bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_ids'],
        min_visibility=0.3,
    )
)
```

#### Why `format='yolo'` matters

Albumentations can work with multiple bbox formats (Pascal VOC, COCO, YOLO, etc.).  
Setting `format='yolo'` tells it bboxes are **(x_center, y_center, width, height)** normalized.

### Why `min_visibility` Matters

`min_visibility=0.3` means:

- If an augmentation (like rotation) pushes most of a box out of the image,
- and less than 30% of that object remains visible,
- Albumentations will **drop that box**.

> **💡 Why dropping is better than keeping**
>
> Keeping boxes for mostly-invisible objects creates bad labels (boxes around “nothing”). That hurts training more than it helps.

---

## Augmentation Engine

### Augment One Image — `augment_single_image`

This function is the “core worker”:

1. Read the image using OpenCV
2. Read YOLO labels
3. Split labels into `class_ids` and `bboxes` (as Albumentations expects)
4. Run augmentation `multiplier` times
5. Save image + save updated YOLO labels

Key pieces:

#### 1) Read the image safely

```python
image = cv2.imread(image_path)
if image is None:
    print(f"  WARNING: Cannot read image: {image_path}")
    return 0
```

#### 2) Prepare bboxes for Albumentations

Albumentations wants:

- `bboxes=[(xc, yc, w, h), ...]`
- `class_ids=[0, 1, 2, ...]`

So the script converts YOLO lines into these two lists.

#### 3) Apply augmentation and save outputs

```python
augmented = pipeline(image=image, bboxes=bboxes, class_ids=class_ids)
aug_image = augmented['image']
aug_bboxes = augmented['bboxes']
aug_class_ids = augmented['class_ids']

aug_name = f"{stem}_aug{start_index + i:03d}"
cv2.imwrite(os.path.join(output_images_dir, f"{aug_name}{ext}"), aug_image)

aug_labels = []
for cls_id, bbox in zip(aug_class_ids, aug_bboxes):
    aug_labels.append([cls_id] + list(bbox))
write_yolo_labels(os.path.join(output_labels_dir, f"{aug_name}.txt"), aug_labels)
```

> **💡 Why `try/except` around each augmentation?**
>
> Some transforms can fail for edge cases (e.g., bbox becomes invalid, image type issues).  
> The script logs the failure and continues so one bad sample doesn’t stop the whole dataset run.

---

### Augment Whole Dataset — `augment_dataset`

This function orchestrates the folder-level operation.

#### 1) Reproducibility: seeds

```python
random.seed(seed)
np.random.seed(seed)
```

> **💡 What this gives you**
>
> With the same seed and same inputs, you usually get the same augmentation sequence (as long as library versions are consistent). This is important for repeatable experiments.

#### 2) Output directories

If you don’t pass output directories, it augments **in-place**:

```python
if output_images_dir is None:
    output_images_dir = images_dir
if output_labels_dir is None:
    output_labels_dir = labels_dir
```

> **⚠️ Practical warning**
>
> In-place augmentation mixes original and augmented files in the same folder. That’s fine if you expect it, but if you re-run augmentation repeatedly, you can end up with runaway dataset growth.

#### 3) Discover image files

It collects images by extension:

```python
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
image_paths.sort()
```

#### 4) Optional copying of originals (only when output differs)

If you set an output folder different from input and `copy_originals=True`, it copies originals too (images + labels):

```python
if copy_originals and output_images_dir != images_dir:
    shutil.copy2(img_path, os.path.join(output_images_dir, Path(img_path).name))
    ...
    shutil.copy2(label_path, os.path.join(output_labels_dir, f"{stem}.txt"))
```

#### 5) Main loop: for each image, find matching label, then augment

```python
for idx, img_path in enumerate(image_paths):
    stem = Path(img_path).stem
    label_path = os.path.join(labels_dir, f"{stem}.txt")

    if not os.path.exists(label_path):
        print(f"  [{idx+1}/{len(image_paths)}] SKIP {stem} — no label file")
        errors += 1
        continue

    n_generated = augment_single_image(..., start_index=idx * multiplier)
    total_augmented += n_generated
```

If a label file is missing, it **skips** that image (and increments `errors`).

---

### Output Naming Strategy

The naming strategy is subtle but important:

```python
start_index = idx * multiplier
aug_name = f"{stem}_aug{start_index + i:03d}"
```

Meaning:

- Image #0 produces: `_aug000` ... `_aug004` (if multiplier=5)
- Image #1 produces: `_aug005` ... `_aug009`
- Image #2 produces: `_aug010` ... `_aug014`

> **💡 Why this is a good idea**
>
> It avoids collisions when the dataset is processed in a single pass and ensures augmented filenames are unique *even if two originals share the same stem in different subfolders* (though this script operates in one folder at a time).

---

## Preview Mode (No Saving)

Preview mode is designed to verify:

- Are augmentations realistic?
- Do bounding boxes still match the defect area?
- Are any transforms too aggressive (especially in `heavy`)?

### Drawing Boxes — `draw_bboxes`

This converts YOLO normalized coordinates into pixel coordinates and draws rectangles:

```python
h, w = image.shape[:2]
xc, yc, bw, bh = bbox
x1 = int((xc - bw / 2) * w)
y1 = int((yc - bh / 2) * h)
x2 = int((xc + bw / 2) * w)
y2 = int((yc + bh / 2) * h)
cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
```

### Preview Loop — `preview_augmentations`

Flow:

1. Randomly sample a few images (`num_samples`)
2. For each sampled image:
  - show original with bboxes
  - show several augmented versions (`augmentations_per_sample`)
3. Wait for keypress to move to next sample (`q` quits)

> **💡 Why preview is useful**
>
> If your dataset has small defects, heavy augmentations (rotate/scale/affine) can make the defect too distorted or too small to learn. Preview lets you tune `level` before generating thousands of files.

---

## CLI (Command Line Interface)

The CLI is in `main()` and supports:

- `--input`: images dir
- `--labels`: labels dir
- `--output-images` / `--output-labels`: output dirs (optional)
- `--multiplier`: copies per image
- `--level`: light/medium/heavy
- `--seed`: reproducibility
- `--preview`: show windows, don’t save
- `--no-copy-originals`: if output dirs differ, don’t copy originals

It enforces Albumentations presence:

```python
if not ALBUMENTATIONS_AVAILABLE:
    print("\nERROR: albumentations is required.")
    ...
    sys.exit(1)
```

And decides between preview vs generation:

- If `--preview` → runs `preview_augmentations(...)`
- Else → runs `augment_dataset(...)`

---

## How to Run

From the project root:

```bash
# Install dependency (if not already installed)
pip install albumentations>=1.3.0

# Augment training set in-place (adds new files into the same folders)
python scripts/augment_dataset.py --input dataset/images/train --labels dataset/labels/train --multiplier 5 --level medium

# Output to a separate folder (recommended if you want clean separation)
python scripts/augment_dataset.py --input dataset/images/train --labels dataset/labels/train --output-images dataset/images/train_aug --output-labels dataset/labels/train_aug --level heavy --multiplier 10

# Preview only (no saving)
python scripts/augment_dataset.py --input dataset/images/train --labels dataset/labels/train --preview
```

> **Windows note (PowerShell)**: If you want multi-line commands, use the backtick continuation character:
>
> ```powershell
> python scripts/augment_dataset.py `
>   --input dataset/images/train `
>   --labels dataset/labels/train `
>   --output-images dataset/images/train_aug `
>   --output-labels dataset/labels/train_aug `
>   --level heavy `
>   --multiplier 10
> ```

---

## How It Connects to Other Scripts

Typical pipeline connection:

```
camera_capture.py            (collect raw images)
       │
       ▼
validate_images.py           (quality checks)
       │
       ▼
prepare_dataset.py           (split/train/val, create YOLO folder layout)
       │
       ▼
augment_dataset.py           (this file: multiply training data)
       │
       ▼
train_model.py               (train YOLO on bigger, more varied dataset)
       │
       ▼
evaluate_model.py / defect_detector.py (validate + run inference)
```

**Best practice**: Augment *only the training set*, not validation/test, so your evaluation remains a fair measure of real-world performance.

---

*This document explains the design and main flow of `augment_dataset.py`. For other scripts, see the corresponding guides in this folder.*  




### **Augmentation techniques used in** `augment_dataset.py`

**Light**

- **Horizontal flip**
- **Random brightness/contrast**
- **Gaussian noise**

**Medium (default)**

- **Horizontal flip**
- **Rotation** (up to ±15°)
- **Random brightness/contrast**
- **OneOf**:
  - **Gaussian noise**
  - **Gaussian blur**
- **CLAHE** (local contrast enhancement)
- **Random shadow**
- **Hue/Saturation/Value shift**

**Heavy**

- **Horizontal flip**
- **Vertical flip**
- **Rotation** (up to ±30°)
- **Random brightness/contrast**
- **OneOf**:
  - **Gaussian noise**
  - **Gaussian blur**
  - **Motion blur**
- **CLAHE**
- **Random shadow**
- **Hue/Saturation/Value shift**
- **Random scale**
- **Piecewise affine distortion**

**BBox-specific behavior**

- Drops boxes that become too occluded after transforms: `min_visibility = 0.3` (keeps only boxes with ≥30% visibility).


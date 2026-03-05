# Model Training Workflow for Defect Detection
## From Image Capture to Production Deployment - Complete Technical Guide

---

## PAGE 1: IMAGE ACQUISITION & DATASET CREATION

### PHASE 1: IMAGE COLLECTION STRATEGY

**1.1 Pre-Collection Planning**

Before capturing a single image, plan your dataset systematically:

**Define Your Classes:**

```
Example Classification Scheme:

Binary Classification (Simple):
├── Class 0: Good (acceptable product)
└── Class 1: Defective (reject)

Multi-Class Classification (Detailed):
├── Class 0: Good
├── Class 1: Scratch
├── Class 2: Crack
├── Class 3: Discoloration
├── Class 4: Dimensional defect
└── Class 5: Contamination

Object Detection (Localized):
├── Background: Good surface
├── BBox Class 1: Scratch (with coordinates)
├── BBox Class 2: Crack (with coordinates)
└── BBox Class 3: Dent (with coordinates)
```

**Determine Dataset Size Requirements:**

| Complexity Level | Minimum Images | Recommended | Optimal |
|-----------------|----------------|-------------|---------|
| Simple binary (2 classes, uniform products) | 500 total | 1,000 | 2,000+ |
| Multi-class (3-5 defect types) | 1,500 total | 3,000 | 5,000+ |
| Object detection (localized defects) | 2,000 total | 5,000 | 10,000+ |
| Complex/variable products | 5,000 total | 10,000 | 20,000+ |

**Class Distribution Strategy:**

```
For Binary Classification:
- Start with 50/50 split (balanced)
- Good products: 500 images
- Defective products: 500 images
- Total: 1,000 images

For Multi-Class (6 classes):
- Target ~150-300 images per class minimum
- Rare defects: Collect more if possible, use augmentation
- Common defects: Balance to match rarest class

Example:
Class 0 (Good): 500 images
Class 1 (Scratch): 200 images
Class 2 (Crack): 150 images
Class 3 (Discoloration): 180 images
Class 4 (Dimensional): 160 images
Class 5 (Contamination): 140 images
Total: 1,330 images
```

**Important:** Production defect rates are often 1-5%, but training datasets should be balanced or slightly imbalanced (not reflecting true distribution). Handle class imbalance during training, not data collection.

---

**1.2 Image Capture Protocol**

**Camera Setup Checklist:**

```
□ Camera Settings:
  - Resolution: _____ × _____ pixels
  - Exposure: _____ ms (fixed, not auto)
  - Gain: _____ dB (fixed, preferably 0 dB)
  - White balance: Manual (if color camera)
  - Focus: Manual, locked
  - Format: RAW or PNG (lossless), NOT JPEG

□ Lighting Setup:
  - Type: LED ring light / Backlight / Dome / Other
  - Intensity: _____ lux (measure with light meter)
  - Color temperature: _____ K (fixed)
  - Angle: _____ degrees
  - Distance: _____ mm from product

□ Product Positioning:
  - Fixture/jig ensures repeatability (±1mm)
  - Background: Neutral (black, white, or consistent)
  - Orientation: Consistent angle
  - Distance from camera: _____ mm (fixed)
```

**Capture Procedure:**

**Step 1: Baseline Good Products**
```
Collect 200-500 "perfect" good products first
- Multiple production batches
- Different time periods (morning/afternoon/night shifts)
- Acceptable color/texture variations
- Document any variations observed
```

**Step 2: Defective Products Collection**

**Real Production Defects:**
```python
# Collection log template
defect_log = {
    'image_id': 'DEF_001',
    'timestamp': '2024-02-06 14:23:15',
    'defect_type': 'scratch',
    'severity': 'major',  # minor / major / critical
    'location': 'top-left',
    'production_batch': 'B-2024-035',
    'operator': 'Station_3',
    'notes': 'Horizontal scratch, 5mm length'
}
```

**If Insufficient Real Defects:**

Option A: **Wait and collect** (slower but better)
Option B: **Simulate defects** (faster but requires validation):
```
Simulated Defects:
- Scratches: Use fine wire, scribing tool
- Cracks: Use thin marker for hairline cracks
- Contamination: Add controlled dust, spots
- Discoloration: Use photo editing (carefully)

WARNING: Simulated defects must be validated 
         against real defects before production use!
```

**Step 3: Diversity Collection**

Capture variations to make model robust:

```
Geometric Variations:
- Rotate product ±15° (if rotation possible in production)
- Slight position shifts (within expected range)
- Different product orientations (if applicable)

Lighting Variations:
- Slight intensity changes (±10%)
- Time of day (if natural light present)
- LED aging simulation (dimmer lighting)

Product Variations:
- Different production batches
- Color variations (within spec)
- Size variations (within tolerance)
- Surface finish variations

Environmental Variations:
- Clean vs slightly dusty lens (realistic)
- Temperature changes (if applicable)
- Different shifts/operators
```

**File Naming Convention:**

```
Structured naming for organization:

Format: {CLASS}_{BATCH}_{SEQUENCE}_{METADATA}.png

Examples:
GOOD_B001_0001_batch123.png
SCRATCH_B002_0056_minor_horizontal.png
CRACK_B003_0023_critical_diagonal.png

Or with timestamp:
20240206_142315_GOOD_0001.png
20240206_143022_SCRATCH_0056.png
```

**Storage Structure:**

```
defect_detection_dataset/
├── raw_images/                    # Original captures (never modify)
│   ├── good/
│   │   ├── GOOD_0001.png
│   │   ├── GOOD_0002.png
│   │   └── ...
│   ├── scratch/
│   │   ├── SCRATCH_0001.png
│   │   └── ...
│   ├── crack/
│   └── discoloration/
│
├── metadata/
│   ├── collection_log.csv         # Detailed capture information
│   └── defect_annotations.json    # Bounding boxes if object detection
│
└── README.md                      # Dataset documentation
```

---

**1.3 Image Quality Validation**

Before annotation, validate image quality:

**Automated Quality Checks:**

```python
import cv2
import numpy as np

def validate_image_quality(image_path):
    """
    Check image quality before adding to dataset
    """
    img = cv2.imread(image_path)
    
    # Check 1: Resolution
    height, width = img.shape[:2]
    if height < 1024 or width < 1024:
        print(f"WARNING: Low resolution {width}x{height}")
    
    # Check 2: Brightness (mean pixel value)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < 50 or mean_brightness > 205:
        print(f"WARNING: Poor brightness {mean_brightness} (target: 100-150)")
    
    # Check 3: Contrast (standard deviation)
    std_contrast = np.std(gray)
    if std_contrast < 20:
        print(f"WARNING: Low contrast {std_contrast}")
    
    # Check 4: Blur detection (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        print(f"WARNING: Image may be blurry {laplacian_var}")
    
    # Check 5: Over/Under exposure (histogram analysis)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    if hist[0] > 0.1 * gray.size or hist[255] > 0.1 * gray.size:
        print("WARNING: Clipping detected (over/under exposure)")
    
    return {
        'resolution': (width, height),
        'brightness': mean_brightness,
        'contrast': std_contrast,
        'sharpness': laplacian_var,
        'passed': True  # Set criteria based on your requirements
    }

# Run on all collected images
import glob
for img_path in glob.glob('raw_images/**/*.png', recursive=True):
    validate_image_quality(img_path)
```

**Manual Quality Review:**

```
Review Checklist (sample 10% of dataset):
□ Product fully visible in frame
□ Consistent lighting across all images
□ No camera artifacts (lens dirt, dead pixels)
□ Defects clearly visible
□ Background consistent
□ Focus sharp (especially at defect locations)
□ No motion blur
□ Correct orientation
```

---

### PHASE 2: DATA ANNOTATION

**2.1 Annotation Strategy by Task Type**

**For Image Classification:**

**Simple folder-based organization:**
```
dataset/
├── train/
│   ├── good/
│   │   ├── img_001.png
│   │   └── img_002.png
│   ├── scratch/
│   └── crack/
├── val/
│   ├── good/
│   ├── scratch/
│   └── crack/
└── test/
    ├── good/
    ├── scratch/
    └── crack/

No additional annotation needed - folder name = label
```

**For Object Detection:**

**YOLO Format (text file per image):**
```
# File: img_001.txt (same name as img_001.png)
# Format: <class_id> <x_center> <y_center> <width> <height>
# Coordinates normalized to 0-1

0 0.5 0.3 0.15 0.08    # Scratch at center-left
1 0.7 0.6 0.12 0.10    # Crack at lower-right

# Where:
# class_id: 0=scratch, 1=crack, 2=dent, etc.
# x_center, y_center: Box center (0.5 = middle of image)
# width, height: Box dimensions (relative to image size)
```

**COCO Format (single JSON file):**
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "img_001.png",
      "width": 2448,
      "height": 2048
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 200, 150, 80],  // [x, y, width, height] in pixels
      "area": 12000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 0, "name": "scratch"},
    {"id": 1, "name": "crack"},
    {"id": 2, "name": "dent"}
  ]
}
```

---

**2.2 Annotation Tools Setup**

**Recommended Tools:**

**Roboflow (Web-based, Easiest):**
```
Pros:
✓ Browser-based, no installation
✓ Built-in augmentation
✓ Export to YOLO, COCO, TensorFlow, PyTorch
✓ Team collaboration
✓ Quality control features
✓ Free tier available

Cons:
✗ Requires internet
✗ Limited free tier (1000 images)

Best for: Small to medium projects, teams
```

**CVAT (Computer Vision Annotation Tool):**
```
Pros:
✓ Open-source, free
✓ Self-hosted or cloud
✓ Supports all annotation types
✓ Video annotation
✓ Multi-user with roles
✓ Interpolation for video

Cons:
✗ Requires setup/installation
✗ Steeper learning curve

Best for: Large projects, on-premise requirements
```

**LabelImg (Simple Desktop App):**
```
Pros:
✓ Lightweight, easy to use
✓ Fast for bounding boxes
✓ YOLO format native support
✓ Keyboard shortcuts

Cons:
✗ Only bounding boxes
✗ Single user
✗ No cloud features

Best for: Quick projects, single annotator
```

**Label Studio:**
```
Pros:
✓ Very flexible (classification, detection, segmentation)
✓ ML-assisted labeling
✓ Web-based
✓ Open-source

Best for: Complex annotation projects
```

---

**2.3 Annotation Best Practices**

**Bounding Box Guidelines:**

```
CORRECT Bounding Box:
┌─────────────────┐
│                 │
│   ┌──────────┐  │
│   │ Scratch  │  │  ← Tight box around defect
│   │   ←──→   │  │     Small margin (few pixels)
│   └──────────┘  │
│                 │
└─────────────────┘

INCORRECT - Too Loose:
┌─────────────────┐
│  ┌──────────┐   │
│  │          │   │  ← Too much background
│  │ Scratch  │   │     Makes model less accurate
│  │          │   │
│  └──────────┘   │
└─────────────────┘

INCORRECT - Partial:
┌─────────────────┐
│        ┌────────│
│ Scratc│        │  ← Missing part of defect
│        └────────│     Model won't learn full pattern
└─────────────────┘
```

**Annotation Consistency Rules:**

```python
# Create annotation guidelines document

annotation_guidelines = """
DEFECT ANNOTATION GUIDELINES

1. SCRATCH:
   - Draw tight box around entire scratch length
   - Include 2-3 pixels margin
   - If multiple scratches close together (<5mm), use single box
   - Minimum length: 2mm (ignore shorter)

2. CRACK:
   - Follow crack path, even if curved
   - Include crack tips
   - For branching cracks, use multiple boxes
   - Minimum visibility: Must be clearly visible

3. DISCOLORATION:
   - Box covers entire discolored region
   - Include gradient edges
   - Minimum area: 5mm²

4. EDGE CASES:
   - Partially visible defect (at edge): Annotate visible portion
   - Unclear defect: Flag for expert review, don't guess
   - Multiple defects overlapping: Separate boxes, allow overlap
   - Very small defects (<1mm): Consider if camera can reliably detect
"""
```

**Quality Control Process:**

```
Annotation QC Workflow:

Step 1: First Pass Annotation
   └─→ Annotator A labels 100 images

Step 2: Cross-Validation (Sample)
   └─→ Annotator B re-labels 10 random images
   └─→ Calculate agreement (IoU > 0.85 = good)

Step 3: Expert Review
   └─→ Domain expert reviews flagged images
   └─→ Creates "golden set" of perfect annotations

Step 4: Consistency Check
   └─→ Automated script checks:
       - All images have labels
       - No boxes outside image bounds
       - No duplicate boxes
       - Class IDs valid

Step 5: Final Audit
   └─→ Review 5% random sample before training
```

**Annotation Agreement Calculation:**

```python
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union for two bounding boxes
    box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# Check agreement between two annotators
annotator_a_box = [100, 100, 200, 150]
annotator_b_box = [105, 98, 205, 148]
iou = calculate_iou(annotator_a_box, annotator_b_box)

if iou > 0.85:
    print("Good agreement")
elif iou > 0.7:
    print("Acceptable, review edge cases")
else:
    print("Poor agreement, re-train annotators")
```

---

**2.4 Dataset Splitting**

**Split Strategy:**

```python
# Standard split ratios
train_split = 0.70  # 70% for training
val_split = 0.15    # 15% for validation (hyperparameter tuning)
test_split = 0.15   # 15% for final testing

# For 1000 images:
# Train: 700 images
# Val: 150 images
# Test: 150 images
```

**Stratified Splitting (Maintain Class Distribution):**

```python
import os
import shutil
from sklearn.model_selection import train_test_split

def stratified_split(dataset_path, output_path, test_size=0.15, val_size=0.15):
    """
    Split dataset while maintaining class distribution
    """
    all_images = []
    all_labels = []
    
    # Collect all image paths and labels
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for img in os.listdir(class_path):
                all_images.append(os.path.join(class_path, img))
                all_labels.append(class_name)
    
    # First split: separate test set
    train_val_imgs, test_imgs, train_val_labels, test_labels = train_test_split(
        all_images, all_labels, 
        test_size=test_size, 
        stratify=all_labels,
        random_state=42
    )
    
    # Second split: separate validation from training
    val_ratio = val_size / (1 - test_size)  # Adjust ratio
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        train_val_imgs, train_val_labels,
        test_size=val_ratio,
        stratify=train_val_labels,
        random_state=42
    )
    
    # Copy files to respective folders
    for split_name, images, labels in [
        ('train', train_imgs, train_labels),
        ('val', val_imgs, val_labels),
        ('test', test_imgs, test_labels)
    ]:
        for img_path, label in zip(images, labels):
            dest_dir = os.path.join(output_path, split_name, label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(img_path, dest_dir)
    
    print(f"Train: {len(train_imgs)} images")
    print(f"Val: {len(val_imgs)} images")
    print(f"Test: {len(test_imgs)} images")

# Usage
stratified_split(
    dataset_path='raw_images/',
    output_path='dataset_split/'
)
```

**Critical Rules for Dataset Splitting:**

```
1. TEMPORAL SPLITTING (Time-based):
   ✓ Test set from LATER batches than training
   ✓ Simulates real deployment (future products)
   
   Example:
   Train: Batches 1-50 (January-March)
   Val: Batches 51-55 (April)
   Test: Batches 56-60 (May)

2. NO DATA LEAKAGE:
   ✗ Same product in train and test
   ✗ Augmented version in test set
   ✗ Similar looking products across splits
   
3. REPRESENTATIVE TEST SET:
   ✓ Test set must represent production variety
   ✓ Include edge cases
   ✓ Include recent production batches

4. VALIDATION SET PURPOSE:
   - Used during training to tune hyperparameters
   - Monitor for overfitting
   - Early stopping trigger
   - NOT for final evaluation (that's test set)
```

---

## PAGE 2: MODEL TRAINING WORKFLOW

### PHASE 3: TRAINING ENVIRONMENT SETUP

**3.1 Hardware Requirements**

```
MINIMUM Setup (for development):
- CPU: 4+ cores
- RAM: 16 GB
- GPU: NVIDIA GTX 1660 (6GB VRAM) or better
- Storage: 256 GB SSD
- Training time: Days to weeks

RECOMMENDED Setup:
- CPU: 8+ cores (Intel i7/i9 or AMD Ryzen 7/9)
- RAM: 32 GB
- GPU: NVIDIA RTX 3060/3070 (12GB VRAM)
- Storage: 512 GB NVMe SSD
- Training time: Hours to days

OPTIMAL Setup:
- CPU: 16+ cores
- RAM: 64 GB
- GPU: NVIDIA RTX 4090 (24GB VRAM) or A100
- Storage: 1 TB NVMe SSD
- Training time: Minutes to hours
```

**Cloud Alternatives:**

```
Google Colab (Free/Pro):
├── Free tier: T4 GPU (15 GB RAM), limited hours
├── Pro ($10/month): Better GPU, longer sessions
└── Best for: Experimentation, small datasets

AWS EC2 (Pay-per-use):
├── p3.2xlarge: V100 GPU, ~$3/hour
├── g4dn.xlarge: T4 GPU, ~$0.50/hour
└── Best for: Production training, large datasets

Paperspace Gradient:
├── Free tier: Limited
├── Pro: RTX 4000/5000, $0.45-0.76/hour
└── Best for: Balance of cost and performance
```

---

**3.2 Software Environment Setup**

**Installation Script (Ubuntu/Linux):**

```bash
#!/bin/bash
# Complete setup script for defect detection training

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10
sudo apt install python3.10 python3.10-venv python3-pip -y

# Install CUDA 11.8 (check NVIDIA website for latest)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Install cuDNN
# Download from NVIDIA website (requires registration)
# Follow installation instructions

# Create virtual environment
python3.10 -m venv defect_detection_env
source defect_detection_env/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Ultralytics (YOLOv8)
pip install ultralytics

# Install additional libraries
pip install opencv-python pandas numpy matplotlib seaborn scikit-learn
pip install albumentations  # Advanced augmentation
pip install tensorboard  # Training visualization
pip install onnx onnxruntime-gpu  # Model export

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

**Python Environment (requirements.txt):**

```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
albumentations>=1.3.0
tensorboard>=2.13.0
onnx>=1.14.0
onnxruntime-gpu>=1.15.0
pillow>=10.0.0
pyyaml>=6.0
tqdm>=4.65.0
```

---

**3.3 Data Augmentation Strategy**

**Why Augmentation:**
```
Benefits:
✓ Increases effective dataset size (500 → 5000+ variations)
✓ Improves model generalization
✓ Reduces overfitting
✓ Simulates real-world variations
✓ Helps with class imbalance
```

**Augmentation Pipeline (using Albumentations):**

```python
import albumentations as A
import cv2

# Define augmentation pipeline
train_transform = A.Compose([
    # Geometric transformations
    A.Rotate(limit=15, p=0.5),  # ±15° rotation, 50% probability
    A.HorizontalFlip(p=0.5),  # Random horizontal flip
    A.VerticalFlip(p=0.3),  # Random vertical flip (less common)
    A.ShiftScaleRotate(
        shift_limit=0.1,  # ±10% shift
        scale_limit=0.15,  # ±15% zoom
        rotate_limit=0,  # Already handled above
        p=0.5
    ),
    
    # Brightness and contrast (simulate lighting changes)
    A.RandomBrightnessContrast(
        brightness_limit=0.2,  # ±20% brightness
        contrast_limit=0.2,  # ±20% contrast
        p=0.6
    ),
    
    # Blur (simulate focus issues)
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
    ], p=0.3),
    
    # Noise (simulate sensor noise, especially important if using gain)
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    
    # Advanced augmentations (use carefully)
    A.CLAHE(clip_limit=2.0, p=0.3),  # Contrast enhancement
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),  # Gamma correction
    
    # Grid distortion (simulates lens distortion)
    A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
    
    # Cutout (randomly mask regions - forces model to use full context)
    A.CoarseDropout(
        max_holes=3, 
        max_height=50, 
        max_width=50, 
        min_holes=1, 
        min_height=20, 
        min_width=20, 
        p=0.3
    ),
])

# Validation/Test: NO augmentation (only resize/normalize)
val_transform = A.Compose([
    A.Resize(640, 640),  # Resize to model input size
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply augmentation
def augment_image(image_path, transform):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    return augmented['image']
```

**Augmentation Best Practices:**

```
DO:
✓ Use augmentation on TRAINING set only
✓ Apply realistic transformations (what could happen in production)
✓ Test augmentations visually before training
✓ Use moderate probabilities (0.3-0.5 for most)
✓ Apply multiple augmentations per image (pipeline)

DON'T:
✗ Augment validation or test sets
✗ Use unrealistic augmentations (extreme distortions)
✗ Over-augment (makes training harder, not better)
✗ Apply augmentations that remove defects
✗ Use augmentation as substitute for real data collection
```

**Defect-Specific Augmentation Considerations:**

```python
# For SCRATCH detection:
# - Rotation: OK (scratches can be any angle)
# - Flip: OK
# - Brightness: OK
# - Heavy blur: NO (may lose thin scratches)

# For COLOR defects (discoloration):
# - Color jitter: Minimal (don't hide defect)
# - Brightness: Minimal
# - Hue shift: NO (changes defect appearance)

# For DIMENSIONAL defects:
# - Perspective transform: NO (changes dimensions)
# - Scale: NO (changes size measurement)
# - Aspect ratio: NO (distorts shape)
```

---

### PHASE 4: MODEL SELECTION & TRAINING

**4.1 Model Architecture Selection**

**Decision Tree:**

```
Q1: Do you need to LOCATE defects (not just detect presence)?
├─ YES → Object Detection (YOLOv8, EfficientDet)
└─ NO → Q2

Q2: Do you have <1000 training images?
├─ YES → Transfer Learning with ResNet/EfficientNet
└─ NO → Q3

Q3: Is inference speed critical (>30 fps needed)?
├─ YES → YOLOv8-nano or MobileNet
└─ NO → YOLOv8-medium or EfficientNet-B3

Q4: Do you need pixel-precise segmentation?
├─ YES → U-Net or Mask R-CNN
└─ NO → Use object detection
```

**Recommended Models by Scenario:**

| Scenario | Recommended Model | Justification |
|----------|------------------|---------------|
| Simple pass/fail, large defects | ResNet-50 (classification) | Fast, accurate, proven |
| Multiple defect types, need location | YOLOv8-small/medium | Best balance speed/accuracy |
| Ultra-fast inspection (>60fps) | YOLOv8-nano | Fastest, acceptable accuracy |
| Very small defects (<1% of image) | YOLOv8-large or EfficientDet | Better small object detection |
| Pixel-precise measurement | U-Net | Precise segmentation |
| Unknown/rare defects | PaDiM (anomaly detection) | No defect examples needed |

---

**4.2 YOLOv8 Training (Most Common Choice)**

**Setup Dataset Configuration:**

```yaml
# defect_dataset.yaml

# Dataset paths
path: /path/to/dataset  # Root directory
train: images/train  # Train images (relative to path)
val: images/val  # Validation images
test: images/test  # Test images (optional)

# Class names
names:
  0: scratch
  1: crack
  2: dent
  3: discoloration
  4: contamination

# Number of classes
nc: 5
```

**Training Script:**

```python
from ultralytics import YOLO
import torch

# Verify GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Load pretrained model (transfer learning)
# Model sizes: yolov8n (nano), yolov8s (small), yolov8m (medium), yolov8l (large), yolov8x (xlarge)
model = YOLO('yolov8s.pt')  # Start with small model

# Train model
results = model.train(
    # Data
    data='defect_dataset.yaml',  # Dataset config file
    
    # Training duration
    epochs=100,  # Number of complete passes through dataset
    patience=20,  # Early stopping after 20 epochs without improvement
    
    # Batch size (adjust based on GPU memory)
    batch=16,  # For 6GB GPU: 8-16, For 12GB: 16-32, For 24GB: 32-64
    
    # Image size
    imgsz=640,  # Input image size (640x640 is standard, use 800 or 1024 for smaller defects)
    
    # Optimization
    optimizer='auto',  # 'SGD', 'Adam', 'AdamW', or 'auto'
    lr0=0.01,  # Initial learning rate
    lrf=0.01,  # Final learning rate (fraction of lr0)
    momentum=0.937,  # SGD momentum
    weight_decay=0.0005,  # Regularization
    
    # Augmentation (YOLOv8 has built-in augmentation)
    degrees=15.0,  # Rotation (±degrees)
    translate=0.1,  # Translation (fraction)
    scale=0.5,  # Scale (±fraction)
    shear=0.0,  # Shear (±degrees)
    perspective=0.0,  # Perspective warp
    flipud=0.0,  # Vertical flip probability
    fliplr=0.5,  # Horizontal flip probability
    mosaic=1.0,  # Mosaic augmentation probability
    mixup=0.1,  # Mixup augmentation probability
    
    # Hardware
    device=0,  # GPU device (0, 1, 2, etc. or 'cpu')
    workers=8,  # Number of worker threads for data loading
    
    # Outputs
    project='defect_detection',  # Project directory
    name='experiment_1',  # Experiment name
    exist_ok=False,  # Overwrite existing project
    save=True,  # Save checkpoints
    save_period=10,  # Save checkpoint every N epochs
    
    # Validation
    val=True,  # Validate during training
    plots=True,  # Save training plots
    
    # Advanced
    amp=True,  # Automatic Mixed Precision (faster training)
    fraction=1.0,  # Use fraction of dataset (1.0 = 100%)
    freeze=None,  # Freeze layers (None or list of layer indices)
)

print("Training complete!")
print(f"Best model: {results.save_dir}/weights/best.pt")
```

**Training Hyperparameters Explained:**

```
EPOCHS:
- Too few (<50): Underfitting, model hasn't learned enough
- Optimal (50-200): Depends on dataset size
- Too many (>300): Overfitting risk, diminishing returns
- Start with 100, use early stopping

BATCH SIZE:
- Larger batch (32, 64): Faster training, more stable gradients, needs more GPU memory
- Smaller batch (8, 16): Slower, more noise, better generalization, less memory
- Rule of thumb: Largest that fits in GPU memory

LEARNING RATE:
- Too high (>0.1): Training unstable, loss jumps
- Too low (<0.0001): Training very slow, may get stuck
- Default (0.01) works for most cases
- Use learning rate scheduler (built into YOLOv8)

IMAGE SIZE:
- 640x640: Standard, good balance
- 800x800 or 1024x1024: Better for small defects, slower training
- 416x416 or 512x512: Faster training, may miss small details
```

---

**4.3 Monitoring Training Progress**

**TensorBoard Visualization:**

```python
# Launch TensorBoard (in separate terminal)
# tensorboard --logdir=defect_detection/experiment_1

# Training metrics to watch:

# 1. LOSS (should decrease):
#    - box_loss: Bounding box localization error
#    - cls_loss: Classification error
#    - dfl_loss: Distribution focal loss
#    - Total loss = sum of above

# 2. METRICS (should increase):
#    - Precision: Of predicted defects, % that are real
#    - Recall: Of actual defects, % that are detected
#    - mAP50: Mean Average Precision at 50% IoU
#    - mAP50-95: mAP averaged from 50% to 95% IoU (stricter)

# 3. LEARNING RATE:
#    - Should decay over time (handled automatically)
```

**Interpreting Training Curves:**

```
GOOD Training:
Loss    │╲
        │ ╲___________  ← Smooth decrease, plateaus
        │
        └────────────── Epochs

Val mAP │      ┌──────  ← Increases, plateaus
        │    ╱
        │  ╱
        └────────────── Epochs


OVERFITTING (BAD):
Train   │╲
Loss    │ ╲____________  ← Train loss decreases
        │
Val     │╲    ╱──────  ← Val loss increases after point
Loss    │ ╲ ╱
        │  V
        └────────────── Epochs
             ↑
        Early stopping point


UNDERFITTING (BAD):
Loss    │╲
        │ ╲
        │  ╲___________  ← Still decreasing, hasn't plateaued
        │
        └────────────── Epochs
        (Need more epochs or larger model)
```

**Early Stopping:**

```python
# YOLOv8 automatically implements early stopping
# If validation mAP doesn't improve for 'patience' epochs, training stops

# Example:
# Epoch 80: mAP = 0.85
# Epoch 90: mAP = 0.87  ← New best!
# Epoch 100: mAP = 0.86
# Epoch 110: mAP = 0.865
# Epoch 120: mAP = 0.864  ← 30 epochs since last improvement
# Training stops, best.pt from epoch 90 is saved
```

---

## PAGE 3: MODEL EVALUATION & DEPLOYMENT

### PHASE 5: MODEL EVALUATION

**5.1 Validation Metrics Deep Dive**

**Confusion Matrix:**

```
For Binary Classification (Good vs Defective):

                  PREDICTED
                Good    Defective
ACTUAL  Good      TN        FP      
        Defective FN        TP

Where:
- TP (True Positive): Correctly identified defective
- TN (True Negative): Correctly identified good
- FP (False Positive): Good marked as defective (waste)
- FN (False Negative): Defective marked as good (quality escape!)

Calculations:
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)  # Of flagged items, % truly defective
Recall = TP / (TP + FN)     # Of defective items, % caught
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

**For Object Detection (mAP):**

```python
# Evaluate YOLOv8 model
from ultralytics import YOLO

model = YOLO('defect_detection/experiment_1/weights/best.pt')

# Run validation
metrics = model.val(data='defect_dataset.yaml')

# Access metrics
print(f"mAP50: {metrics.box.map50}")  # mAP at IoU=0.5
print(f"mAP50-95: {metrics.box.map}")  # mAP at IoU=0.5:0.95
print(f"Precision: {metrics.box.mp}")
print(f"Recall: {metrics.box.mr}")

# Per-class AP
for i, class_name in enumerate(model.names.values()):
    print(f"{class_name}: AP50 = {metrics.box.ap50[i]}")
```

**Understanding mAP:**

```
mAP (mean Average Precision) combines:
1. Precision-Recall curve for each class
2. Area under PR curve = Average Precision (AP)
3. Mean of all class APs = mAP

mAP50: IoU threshold = 0.5 (box overlap ≥50% counts as correct)
mAP50-95: Average of mAP at IoU thresholds 0.5, 0.55, 0.6, ... 0.95

Target Values:
- mAP50 > 0.90: Excellent
- mAP50 > 0.80: Good
- mAP50 > 0.70: Acceptable for some applications
- mAP50 < 0.70: Needs improvement

Production Quality:
- Recall > 0.95: Catch 95%+ of defects (critical!)
- Precision > 0.90: Keep false alarms low
```

---

**5.2 Test Set Evaluation**

```python
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('best.pt')

# Predict on test set
results = model.predict(
    source='dataset/test/images',
    conf=0.5,  # Confidence threshold
    iou=0.45,  # NMS IoU threshold
    save=True,  # Save annotated images
    save_txt=True,  # Save predictions as txt
)

# Calculate business metrics
total_products = 0
true_defects = 0
detected_defects = 0
false_positives = 0
false_negatives = 0

for result in results:
    total_products += 1
    
    # Ground truth (from label file)
    label_path = result.path.replace('images', 'labels').replace('.png', '.txt')
    has_defect = os.path.exists(label_path) and os.path.getsize(label_path) > 0
    
    # Prediction
    detected = len(result.boxes) > 0
    
    if has_defect:
        true_defects += 1
        if detected:
            detected_defects += 1
        else:
            false_negatives += 1
    else:
        if detected:
            false_positives += 1

# Calculate rates
defect_rate = true_defects / total_products
catch_rate = detected_defects / true_defects if true_defects > 0 else 0
false_positive_rate = false_positives / (total_products - true_defects)

print(f"Total products tested: {total_products}")
print(f"Actual defect rate: {defect_rate:.2%}")
print(f"Catch rate (Recall): {catch_rate:.2%}")
print(f"False positive rate: {false_positive_rate:.2%}")
print(f"False negatives (missed defects): {false_negatives}")
```

**Business-Focused Metrics:**

```
Cost of False Negative (Missed Defect):
- Defective product shipped to customer
- Potential return, warranty claim, reputation damage
- Cost: $50 - $10,000+ depending on product

Cost of False Positive (Good Product Rejected):
- Wasted product (scrap cost)
- Reduced yield
- Manual review time
- Cost: $5 - $100 per product

Optimization:
If FN cost >> FP cost → Lower confidence threshold (catch more, accept more false alarms)
If FP cost >> FN cost → Raise confidence threshold (reject less, risk missing some)

Example:
FN cost = $500, FP cost = $10
→ Accept 50 false positives to avoid 1 false negative
→ Use lower confidence threshold (0.3-0.4)
```

---

**5.3 Error Analysis**

```python
# Analyze false positives and false negatives
from collections import defaultdict

error_analysis = defaultdict(list)

for result in results:
    # Compare prediction vs ground truth
    pred_boxes = result.boxes.xyxy.cpu().numpy()
    pred_classes = result.boxes.cls.cpu().numpy()
    pred_confs = result.boxes.conf.cpu().numpy()
    
    # Load ground truth
    label_path = result.path.replace('images', 'labels').replace('.png', '.txt')
    
    if os.path.exists(label_path):
        gt_boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                # Parse YOLO format
                parts = line.strip().split()
                # ... convert to pixel coordinates
                gt_boxes.append(parts)
        
        # Match predictions to ground truth
        # Find false negatives (missed defects)
        # Find false positives (incorrect detections)
        # Categorize errors
    
# Common error patterns:
print("Error Analysis:")
print(f"1. Small defects missed: {count}")
print(f"2. Edge defects missed: {count}")
print(f"3. Low contrast defects missed: {count}")
print(f"4. False alarms on normal features: {count}")
print(f"5. Multiple detections on single defect: {count}")

# Visualize errors
# Save images of all false positives and false negatives for review
```

**Common Issues and Solutions:**

```
ISSUE: Model misses small defects
SOLUTION:
- Increase input image size (640 → 800 or 1024)
- Use larger model (yolov8m → yolov8l)
- Collect more examples of small defects
- Improve lighting to increase defect contrast

ISSUE: High false positive rate on normal features
SOLUTION:
- Collect more diverse "good" examples
- Review annotations (are they truly defects?)
- Increase confidence threshold
- Add hard negative mining (collect false positives, label as good)

ISSUE: Model performs well in training but poor in production
SOLUTION:
- Collect more diverse training data (different lighting, batches, etc.)
- Test on recent production samples before deployment
- Implement continuous learning (retrain with production data)

ISSUE: Inconsistent results (same product gives different predictions)
SOLUTION:
- Fix camera settings (disable auto-exposure, auto-gain)
- Improve product positioning repeatability
- Add more robust augmentation during training
```

---

### PHASE 6: MODEL OPTIMIZATION FOR DEPLOYMENT

**6.1 Model Export and Optimization**

**Export to ONNX (for cross-platform deployment):**

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('best.pt')

# Export to ONNX
model.export(
    format='onnx',  # Export format
    imgsz=640,  # Input size
    dynamic=False,  # Fixed input size (faster)
    simplify=True,  # Simplify model graph
    opset=12,  # ONNX opset version
)

print("Model exported to best.onnx")
```

**Quantization (for edge devices):**

```python
# INT8 Quantization (4x smaller, 2-4x faster)
# Requires representative calibration data

import torch
from ultralytics import YOLO

model = YOLO('best.pt')

# For Jetson/embedded: Export to TensorRT
model.export(
    format='engine',  # TensorRT engine
    imgsz=640,
    half=True,  # FP16 precision
    workspace=4,  # Max workspace size (GB)
    int8=True,  # INT8 quantization
    data='defect_dataset.yaml',  # For calibration
)

# For other platforms: ONNX with quantization
model.export(format='onnx', simplify=True)

# Then use onnxruntime with quantization
# pip install onnxruntime-gpu
```

**Model Pruning (reduce size and latency):**

```python
# Remove unnecessary model weights
# Advanced technique, use with caution

import torch
import torch.nn.utils.prune as prune

# Load model
model = YOLO('best.pt').model

# Apply pruning to convolutional layers
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.3)  # Remove 30% of weights
        prune.remove(module, 'weight')  # Make pruning permanent

# Re-train briefly to recover accuracy (fine-tuning)
# Save pruned model
```

---

**6.2 Inference Speed Benchmarking**

```python
import time
import numpy as np
from ultralytics import YOLO

model = YOLO('best.onnx')  # Or best.pt, best.engine

# Warmup (first inference is slow)
for _ in range(10):
    model('test_image.png')

# Benchmark
num_iterations = 100
times = []

for _ in range(num_iterations):
    start = time.time()
    result = model('test_image.png', verbose=False)
    end = time.time()
    times.append((end - start) * 1000)  # Convert to ms

# Statistics
avg_time = np.mean(times)
std_time = np.std(times)
fps = 1000 / avg_time

print(f"Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
print(f"Throughput: {fps:.1f} FPS")
print(f"Min: {np.min(times):.2f} ms, Max: {np.max(times):.2f} ms")

# Check if meets production requirements
required_fps = 30  # Example requirement
if fps >= required_fps:
    print(f"✓ Meets requirement ({required_fps} FPS)")
else:
    print(f"✗ Too slow. Need {required_fps - fps:.1f} FPS improvement")
    print("Suggestions: Use smaller model, reduce image size, or upgrade hardware")
```

**Optimization Comparison:**

```
PERFORMANCE COMPARISON (YOLOv8s, 640x640, RTX 3060):

Format          | Size   | Inference Time | FPS | Relative Speed
----------------|--------|----------------|-----|---------------
PyTorch (.pt)   | 22 MB  | 15 ms          | 67  | 1.0x (baseline)
ONNX (.onnx)    | 22 MB  | 12 ms          | 83  | 1.25x
TensorRT FP16   | 11 MB  | 6 ms           | 167 | 2.5x
TensorRT INT8   | 6 MB   | 4 ms           | 250 | 3.75x

Note: INT8 may reduce accuracy by 1-3% mAP
```

---

**6.3 Deployment Architecture**

**Simple Standalone Deployment:**

```python
# inference.py - Production inference script

import cv2
from ultralytics import YOLO
import time

class DefectDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        """Initialize defect detection model"""
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
    def inspect(self, image_path):
        """
        Run defect detection on single image
        
        Returns:
            dict: {
                'is_defective': bool,
                'defects': list of {class, confidence, bbox},
                'inference_time_ms': float
            }
        """
        start = time.time()
        
        # Run inference
        results = self.model(image_path, conf=self.conf_threshold, verbose=False)
        
        inference_time = (time.time() - start) * 1000
        
        # Parse results
        defects = []
        for result in results:
            for box in result.boxes:
                defects.append({
                    'class': self.model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy.tolist()[0]  # [x1, y1, x2, y2]
                })
        
        return {
            'is_defective': len(defects) > 0,
            'defects': defects,
            'inference_time_ms': inference_time
        }

# Usage
if __name__ == '__main__':
    detector = DefectDetector('best.onnx', conf_threshold=0.6)
    
    # Inspect product
    result = detector.inspect('product_image.png')
    
    if result['is_defective']:
        print(f"REJECT - Found {len(result['defects'])} defect(s)")
        for defect in result['defects']:
            print(f"  - {defect['class']}: {defect['confidence']:.2%}")
    else:
        print("PASS - No defects detected")
    
    print(f"Inference time: {result['inference_time_ms']:.1f} ms")
```

**REST API Deployment (for networked systems):**

```python
# api_server.py - FastAPI server for remote inference

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from inference import DefectDetector
import io

app = FastAPI(title="Defect Detection API")
detector = DefectDetector('best.onnx', conf_threshold=0.5)

@app.post("/inspect")
async def inspect_product(file: UploadFile = File(...)):
    """
    Endpoint for defect inspection
    
    Args:
        file: Image file (PNG, JPG)
    
    Returns:
        JSON with inspection results
    """
    # Read image from upload
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Save temporarily
    temp_path = '/tmp/temp_inspection.png'
    cv2.imwrite(temp_path, img)
    
    # Run inspection
    result = detector.inspect(temp_path)
    
    return JSONResponse(content=result)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True}

# Run with: uvicorn api_server:app --host 0.0.0.0 --port 8000
```

**Client-side Integration:**

```python
# client.py - Call API from inspection station

import requests

def inspect_via_api(image_path, api_url="http://192.168.1.100:8000"):
    """Send image to API for inspection"""
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{api_url}/inspect", files=files, timeout=5)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API error: {response.status_code}")

# Usage
result = inspect_via_api('camera_capture.png')
if result['is_defective']:
    # Trigger reject mechanism
    activate_reject_pusher()
```

---

**6.4 Production Monitoring and Logging**

```python
# logger.py - Log all inspections to database

import sqlite3
from datetime import datetime
import json

class InspectionLogger:
    def __init__(self, db_path='inspections.db'):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create database schema"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_path TEXT,
                result TEXT,  -- 'PASS' or 'REJECT'
                defects TEXT,  -- JSON string
                confidence REAL,
                inference_time_ms REAL,
                operator_shift TEXT,
                production_batch TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def log_inspection(self, image_path, result, metadata=None):
        """Log inspection result"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            INSERT INTO inspections 
            (timestamp, image_path, result, defects, confidence, inference_time_ms, operator_shift, production_batch)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            image_path,
            'REJECT' if result['is_defective'] else 'PASS',
            json.dumps(result['defects']),
            max([d['confidence'] for d in result['defects']]) if result['defects'] else 0,
            result['inference_time_ms'],
            metadata.get('shift', 'unknown'),
            metadata.get('batch', 'unknown')
        ))
        
        conn.commit()
        conn.close()
    
    def get_stats(self, start_date=None, end_date=None):
        """Get inspection statistics"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT result, COUNT(*) FROM inspections"
        if start_date:
            query += f" WHERE timestamp >= '{start_date}'"
        query += " GROUP BY result"
        
        results = conn.execute(query).fetchall()
        conn.close()
        
        stats = {result: count for result, count in results}
        total = sum(stats.values())
        
        return {
            'total_inspections': total,
            'passed': stats.get('PASS', 0),
            'rejected': stats.get('REJECT', 0),
            'defect_rate': stats.get('REJECT', 0) / total if total > 0 else 0
        }

# Usage
logger = InspectionLogger()
result = detector.inspect('product.png')
logger.log_inspection('product.png', result, metadata={'shift': 'A', 'batch': 'B-123'})

# Get daily stats
stats = logger.get_stats(start_date='2024-02-06')
print(f"Defect rate: {stats['defect_rate']:.2%}")
```

---

**6.5 Continuous Improvement Loop**

```python
# continuous_learning.py - Collect edge cases for retraining

import shutil
from pathlib import Path

class ContinuousLearning:
    def __init__(self, 
                 low_confidence_threshold=0.7,
                 collection_dir='edge_cases'):
        self.low_conf_threshold = low_confidence_threshold
        self.collection_dir = Path(collection_dir)
        self.collection_dir.mkdir(exist_ok=True)
        
        (self.collection_dir / 'uncertain').mkdir(exist_ok=True)
        (self.collection_dir / 'manual_review').mkdir(exist_ok=True)
    
    def collect_edge_case(self, image_path, result):
        """
        Collect low-confidence predictions for review and retraining
        """
        if not result['defects']:
            return False
        
        max_conf = max([d['confidence'] for d in result['defects']])
        
        if max_conf < self.low_conf_threshold:
            # Copy to edge cases directory
            dest = self.collection_dir / 'uncertain' / Path(image_path).name
            shutil.copy(image_path, dest)
            
            # Log for manual review
            with open(self.collection_dir / 'review_queue.txt', 'a') as f:
                f.write(f"{dest},{max_conf},{datetime.now().isoformat()}\n")
            
            return True
        
        return False
    
    def prepare_retraining_dataset(self, reviewed_labels_path):
        """
        Combine original dataset with manually labeled edge cases
        for model retraining
        """
        # Implementation: Merge datasets, re-split, trigger training pipeline
        pass

# Usage
cl = ContinuousLearning()
result = detector.inspect('product.png')

if cl.collect_edge_case('product.png', result):
    print("Low confidence detection - flagged for manual review")
    # Route to manual inspection
```

---

### SUMMARY: COMPLETE WORKFLOW CHECKLIST

```
□ PHASE 1: Image Collection
  □ Define classes and target dataset size
  □ Set up camera with fixed settings
  □ Collect balanced dataset
  □ Validate image quality

□ PHASE 2: Annotation
  □ Choose annotation tool
  □ Create annotation guidelines
  □ Annotate all images
  □ Quality control review
  □ Split dataset (70/15/15)

□ PHASE 3: Training Setup
  □ Set up hardware/environment
  □ Install dependencies
  □ Configure augmentation pipeline

□ PHASE 4: Model Training
  □ Select model architecture
  □ Configure training parameters
  □ Monitor training progress
  □ Implement early stopping

□ PHASE 5: Evaluation
  □ Validate on test set
  □ Calculate business metrics
  □ Perform error analysis
  □ Iterate if needed

□ PHASE 6: Deployment
  □ Export and optimize model
  □ Benchmark inference speed
  □ Integrate with production system
  □ Set up logging and monitoring
  □ Implement continuous learning

□ ONGOING: Maintenance
  □ Monitor daily performance
  □ Collect edge cases
  □ Periodic retraining (monthly/quarterly)
  □ Update model as products evolve
```

**Your model is now ready for production deployment!**

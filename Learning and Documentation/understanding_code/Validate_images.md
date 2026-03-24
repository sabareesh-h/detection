# 🔍 Validate Images — Complete Code Walkthrough
**In our defect detection pipeline,

`validate_images.py`
 is the "Quality Gate" of the system. It sits between the camera (capturing images) and the training pipeline. Without it, you'd feed garbage data into your AI model — blurry photos, dark images, over-exposed shots — and get a terrible model back.

Here are the 4 main reasons why we need this specific file:

1. **Garbage In, Garbage Out (Data Quality)**
AI models learn from data. If you train YOLO on 100 blurry photos, it will learn that "blurry" is normal — and miss sharp defects. This script checks every image BEFORE it enters the training pipeline.

2. **Automated Quality Control (Speed)**
Manually checking 500 images for quality is slow and error-prone. This script checks resolution, brightness, contrast, sharpness, and exposure in milliseconds — and gives you a report.

3. **Objective Measurements (Consistency)**
"This image looks okay" is subjective. This script uses mathematical metrics (Laplacian variance for sharpness, pixel histogram for exposure) that give the same result every time.

4. **Early Problem Detection (Cost Savings)**
Training a model takes hours. This script catches bad images in seconds — before you waste GPU time training on poor data.
**

> **Script**: `scripts/validate_images.py`  
> **Purpose**: Validate image quality (resolution, brightness, contrast, sharpness, exposure) before adding images to the training dataset.  
> **When to use**: After capturing images with `camera_capture.py` — Step 2 of the pipeline.

---

## Table of Contents

- [Overview](#overview)
- [Imports & Dependencies](#imports--dependencies)
- [ImageQualityValidator Class](#imagequalityvalidator-class)
  - [Initialization — `__init__`](#initialization--__init__)
  - [Validating a Single Image — `validate`](#validating-a-single-image--validate)
  - [Validating a Directory — `validate_directory`](#validating-a-directory--validate_directory)
  - [Printing the Report — `print_report`](#printing-the-report--print_report)
- [Main Function — `main()`](#main-function--main)
- [Script Entry Point](#script-entry-point)
- [How to Run](#how-to-run)
- [How It Connects to Other Scripts](#how-it-connects-to-other-scripts)

---

## Overview

This script has **2 main components**:

```
┌───────────────────────────────────────────────────────┐
│                  validate_images.py                    │
│                                                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │          ImageQualityValidator (class)           │  │
│  │                                                  │  │
│  │  ┌───────────┐  ┌──────────────┐  ┌──────────┐ │  │
│  │  │ validate  │  │validate_     │  │print_    │ │  │
│  │  │(1 image)  │  │directory()   │  │report()  │ │  │
│  │  │           │  │(all images)  │  │(summary) │ │  │
│  │  └─────┬─────┘  └──────┬───────┘  └────┬─────┘ │  │
│  │        │               │                │       │  │
│  │        │   Checks 5 quality metrics:    │       │  │
│  │        │   1. Resolution                │       │  │
│  │        │   2. Brightness                │       │  │
│  │        │   3. Contrast                  │       │  │
│  │        │   4. Sharpness                 │       │  │
│  │        │   5. Exposure clipping         │       │  │
│  └────────┴────────────────────────────────┴───────┘  │
│                                                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │              main() function                     │  │
│  │  Parses arguments, creates validator, runs it    │  │
│  └─────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────┘
```

1. **`ImageQualityValidator`** — The core class that checks image quality with 5 automated tests
2. **`main()`** — Command-line interface that parses arguments and runs the validator

---

## Imports & Dependencies

```python
"""                                                  # Lines 1-4
Image Quality Validation Script                      # Docstring — describes the script's purpose
Validates captured images before adding to           # This runs AFTER camera_capture.py
training dataset                                     # and BEFORE training
"""

import os                                            # Line 6: File system operations
import glob                                          # Line 7: Find files matching patterns (*.png, *.jpg)
import cv2                                           # Line 8: OpenCV — image loading & processing
import numpy as np                                   # Line 9: NumPy — mathematical operations on images
from pathlib import Path                             # Line 10: Modern/clean file path handling
from typing import Dict, List, Tuple                 # Line 11: Type hints for function signatures
```

> **💡 What's new compared to camera_capture.py?**
>
> | Import | Purpose | New Here? |
> |--------|---------|-----------|
> | `glob` | Find files matching patterns like `*.png` recursively | ✅ Yes |
> | `typing` | Type annotations (`Dict`, `List`, `Tuple`) for code clarity | ✅ Yes |
> | `cv2` | Image loading and analysis | Same as camera_capture.py |
> | `numpy` | Math on pixel arrays | Same as camera_capture.py |
>
> **Why `glob` instead of `os.listdir()`?**  
> `glob.glob("**/*.png", recursive=True)` finds all `.png` files in ALL subdirectories in one line. With `os.listdir()` you'd need to write a recursive loop yourself. `glob` supports wildcard patterns like `*` (any filename) and `**` (any depth of subdirectories).
>
> **Why `typing`?**  
> Type hints like `Tuple[int, int]` tell other developers (and your IDE) what types a function expects. Python doesn't enforce them at runtime — they're documentation that your IDE can check for errors.

---

## ImageQualityValidator Class

### Initialization — `__init__`

```python
class ImageQualityValidator:                         # Line 14
    """Validates image quality for defect             # Line 15: Class docstring
    detection training"""

    def __init__(self,                               # Line 17: Constructor
                 min_resolution: Tuple[int, int]      # Line 18: Type hint → expects (width, height)
                     = (640, 480),                    #           Default: at least 640×480 pixels
                 brightness_range: Tuple[int, int]    # Line 19: (min_brightness, max_brightness)
                     = (50, 205),                     #           Acceptable range: 50–205
                 min_contrast: float = 20.0,          # Line 20: Minimum contrast threshold
                 min_sharpness: float = 100.0):       # Line 21: Minimum sharpness threshold
```

> **💡 What do these default values mean?**
>
> | Parameter | Default | What It Controls |
> |-----------|---------|------------------|
> | `min_resolution` | `(640, 480)` | Images must be at least 640 pixels wide and 480 tall. Smaller = not enough detail for YOLO |
> | `brightness_range` | `(50, 205)` | Mean pixel brightness must be between 50 (not too dark) and 205 (not too bright). Range is 0–255 |
> | `min_contrast` | `20.0` | Standard deviation of pixel values must be ≥ 20. Low contrast = everything looks the same shade |
> | `min_sharpness` | `100.0` | Laplacian variance must be ≥ 100. Low value = blurry image |
>
> **Why these specific values?**  
> These are tuned for industrial defect detection. A mean brightness of 50 means the image is quite dark (imagine a photo taken in a dim room). 205 means very bright (almost washed out). The range 50–205 gives a comfortable middle ground where defects are visible.

```python
        """                                          # Lines 22-30: Docstring
        Initialize validator with quality thresholds

        Args:
            min_resolution: Minimum (width, height)
            brightness_range: Acceptable (min, max) mean brightness
            min_contrast: Minimum standard deviation for contrast
            min_sharpness: Minimum Laplacian variance for sharpness
        """
        self.min_resolution = min_resolution         # Line 31: Store as instance variable
        self.brightness_range = brightness_range     # Line 32: Store as instance variable
        self.min_contrast = min_contrast             # Line 33: Store as instance variable
        self.min_sharpness = min_sharpness           # Line 34: Store as instance variable
```

> **💡 Why store as `self.xxx`?**
>
> `self.min_resolution` makes the value available in ALL methods of the class (like `validate()`, `validate_directory()`). Without `self.`, the variable would be local to `__init__` and lost after initialization finishes.

---

### Validating a Single Image — `validate`

This is the **core method** — it runs 5 quality checks on a single image:

```python
    def validate(self, image_path: str) -> Dict:     # Line 36: Takes filepath, returns a dict
        """                                          # Lines 37-44: Docstring
        Validate a single image

        Args:
            image_path: Path to image file

        Returns:
            dict with validation results and metrics
        """
        img = cv2.imread(image_path)                 # Line 46: Load image from disk
```

> **💡 What does `cv2.imread()` return?**
>
> - **Success**: A NumPy array with shape `(height, width, 3)` — the 3 channels are Blue, Green, Red (BGR)
> - **Failure**: `None` — if the file doesn't exist, is corrupted, or isn't an image format
>
> This is why the next line checks for `None`.

```python
        if img is None:                              # Line 48: Image failed to load?
            return {                                 # Line 49: Return a failure dict
                'path': image_path,                  # Line 50: Which file failed
                'passed': False,                     # Line 51: Mark as failed
                'error': 'Failed to load image',    # Line 52: Error description
                'issues': ['Cannot read image file'] # Line 53: List of problems
            }
```

> **💡 When does `cv2.imread()` return `None`?**
>
> - The file doesn't exist (wrong path)
> - The file is corrupted (partially downloaded, disk error)
> - The file isn't actually an image (e.g., a renamed `.txt` file with `.png` extension)
> - Unsupported format (e.g., `.webp` without the right codec)

```python
        issues = []                                  # Line 56: Empty list — we'll add problems here
        metrics = {}                                 # Line 57: Empty dict — we'll add measurements
```

> **💡 Design pattern note:**
>
> This is the "accumulator" pattern. We start with empty collections and add items as we find issues. At the end, `len(issues) == 0` means the image passed all checks. This is cleaner than using a boolean `passed = True` and setting it to `False` — because we also collect the *reasons* why it failed.

---

#### Check 1: Resolution

```python
        # Check 1: Resolution
        height, width = img.shape[:2]                # Line 60: Get image dimensions
        metrics['resolution'] = (width, height)      # Line 61: Record the actual resolution

        if width < self.min_resolution[0] or \
           height < self.min_resolution[1]:           # Line 62: Compare to minimum
            issues.append(                           # Line 63: Add issue if too small
                f"Low resolution: {width}x{height} "
                f"(min: {self.min_resolution[0]}x"
                f"{self.min_resolution[1]})"
            )
```

> **💡 Why `img.shape[:2]` and not `img.shape`?**
>
> `img.shape` returns `(height, width, channels)` — e.g., `(480, 640, 3)`.  
> `img.shape[:2]` slices off the last value, giving just `(height, width)` — `(480, 640)`.  
> We don't need the channel count here — we only care about pixel dimensions.
>
> **⚠️ Note**: OpenCV stores shape as `(height, width)` — NOT `(width, height)`. This is because images are 2D arrays where rows = height and columns = width. It trips up many beginners!
>
> **Why does resolution matter for YOLO?**  
> YOLO resizes all input images to 640×640. If your source image is only 100×100, it gets stretched 6.4× — making everything blurry and pixelated. Defects that are 5 pixels wide become undetectable.

---

#### Check 2: Brightness

```python
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
            if len(img.shape) == 3 else img          # Line 66: Convert to grayscale (if color)
```

> **💡 Why convert to grayscale?**
>
> Brightness, contrast, and sharpness are all single-value metrics. A color image has 3 channels (B, G, R). By converting to grayscale (1 channel), we get one intensity value per pixel, which makes the math straightforward.
>
> The `if len(img.shape) == 3` check handles the case where the image is *already* grayscale (shape would be `(480, 640)` with only 2 dimensions). This defensive check prevents a crash.

```python
        # Check 2: Brightness (mean pixel value)
        mean_brightness = np.mean(gray)              # Line 69: Average of ALL pixel values
        metrics['brightness'] = round(               # Line 70: Record the value
            mean_brightness, 2)                      #           Round to 2 decimal places

        if mean_brightness < self.brightness_range[0]:  # Line 71: Too dark?
            issues.append(
                f"Too dark: brightness={mean_brightness:.1f} "
                f"(min: {self.brightness_range[0]})"
            )
        elif mean_brightness > self.brightness_range[1]:  # Line 73: Too bright?
            issues.append(
                f"Too bright: brightness={mean_brightness:.1f} "
                f"(max: {self.brightness_range[1]})"
            )
```

> **💡 Understanding brightness as a number:**
>
> ```
> Pixel values:  0 ──────────────── 127 ──────────────── 255
>                │                    │                     │
>             Pure Black          Mid Gray            Pure White
>
> Our range:    [50] ──── Acceptable ──── [205]
>                │                          │
>            "Too dark"                "Too bright"
> ```
>
> `np.mean(gray)` adds up every pixel value in the image and divides by the total number of pixels. For a 640×480 image, that's averaging **307,200 values** into one number.
>
> | Mean Brightness | Visual Appearance | Verdict |
> |------|---------|---------|
> | 10 | Almost pitch black | ❌ Too dark |
> | 50 | Very dim | ✅ Just passes |
> | 127 | Middle gray | ✅ Ideal |
> | 205 | Very bright | ✅ Just passes |
> | 240 | Almost white | ❌ Too bright |

---

#### Check 3: Contrast

```python
        # Check 3: Contrast (standard deviation)
        std_contrast = np.std(gray)                  # Line 77: Standard deviation of pixel values
        metrics['contrast'] = round(std_contrast, 2) # Line 78: Record the value

        if std_contrast < self.min_contrast:          # Line 79: Too low?
            issues.append(
                f"Low contrast: std={std_contrast:.1f} "
                f"(min: {self.min_contrast})"
            )
```

> **💡 What is contrast and why use standard deviation?**
>
> **Contrast** = how much variation there is in the brightness of an image.
>
> ```
> LOW CONTRAST (std ≈ 5):              HIGH CONTRAST (std ≈ 60):
> ┌──────────────────┐                 ┌──────────────────┐
> │ ░░░░░░░░░░░░░░░░ │                 │ ██░░░░████░░░░██ │
> │ ░░░░░░░░░░░░░░░░ │  Everything     │ ░░████░░░░████░░ │  Clear difference
> │ ░░░░░░░░░░░░░░░░ │  looks the      │ ████░░░░░░░░████ │  between light
> │ ░░░░░░░░░░░░░░░░ │  same shade     │ ░░░░████████░░░░ │  and dark areas
> └──────────────────┘                 └──────────────────┘
> ```
>
> **Standard deviation** measures spread: if all pixels are the same value (solid gray), std = 0. If pixels range from pure black to pure white, std is high.
>
> **Why does low contrast fail?**  
> If everything is the same shade, defects (scratches, cracks) won't be visible to the model. Good contrast means defects stand out from the background.
>
> `np.std(gray)` computes the standard deviation across all pixel values — it tells you how "spread out" the brightness values are.

---

#### Check 4: Sharpness (Blur Detection)

```python
        # Check 4: Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(               # Line 83: Apply Laplacian filter
            gray, cv2.CV_64F                         #   cv2.CV_64F = 64-bit float output
        ).var()                                      #   .var() = variance of the result
        metrics['sharpness'] = round(laplacian_var, 2)  # Line 84: Record the value

        if laplacian_var < self.min_sharpness:        # Line 85: Too blurry?
            issues.append(
                f"Blurry: sharpness={laplacian_var:.1f} "
                f"(min: {self.min_sharpness})"
            )
```

> **💡 How does the Laplacian detect blur? (The key insight)**
>
> The **Laplacian** is a mathematical filter that detects **edges** (sudden changes in brightness). It looks at each pixel and asks: "How different is this pixel from its neighbors?"
>
> ```
> Sharp image:                    Blurry image:
> Pixel values:                   Pixel values:
> [10, 10, 200, 200, 10]         [10, 50, 120, 150, 100]
>       ↑                              ↑
>  Sharp edge!                    Gradual change...
>  Laplacian = HIGH               Laplacian = LOW
> ```
>
> **Step by step:**
> 1. `cv2.Laplacian(gray, cv2.CV_64F)` — Applies the Laplacian filter to every pixel. Sharp edges produce large values; smooth areas produce near-zero values.
> 2. `.var()` — Computes the **variance** (spread) of all those values. If the image has many sharp edges, the variance is high. If the image is uniformly blurry, the variance is low.
>
> **Why `cv2.CV_64F`?**  
> The Laplacian can produce negative values (when going from bright to dark). `CV_64F` (64-bit float) can handle negative numbers. If we used `CV_8U` (8-bit unsigned), negatives would clip to 0 and we'd lose information.
>
> | Laplacian Variance | Meaning | Example |
> |----|---------|---------|
> | < 50 | Very blurry | Camera out of focus |
> | 100 | Borderline | Slight motion blur |
> | 500+ | Very sharp | Well-focused image |
> | 2000+ | Extremely sharp | Macro/close-up with fine detail |

---

#### Check 5: Exposure Clipping

```python
        # Check 5: Clipping (over/under exposure)
        hist = cv2.calcHist(                         # Line 89: Calculate brightness histogram
            [gray],                                  #   Input: grayscale image (as list)
            [0],                                     #   Channel index: 0 (only one for grayscale)
            None,                                    #   Mask: None = use entire image
            [256],                                   #   Number of bins: 256 (one per brightness level)
            [0, 256]                                 #   Range: 0 to 256 (all possible values)
        )
        pixel_count = gray.size                      # Line 90: Total number of pixels
        underexposed = hist[0][0] / pixel_count      # Line 91: % of pixels that are pure black (0)
        overexposed = hist[255][0] / pixel_count     # Line 92: % of pixels that are pure white (255)
        metrics['underexposed_pct'] = round(         # Line 93: Record as percentage
            underexposed * 100, 2)
        metrics['overexposed_pct'] = round(          # Line 94: Record as percentage
            overexposed * 100, 2)

        if underexposed > 0.1:                       # Line 96: >10% pixels pure black?
            issues.append(
                f"Underexposed: {underexposed*100:.1f}% "
                f"pixels clipped at black"
            )
        if overexposed > 0.1:                        # Line 98: >10% pixels pure white?
            issues.append(
                f"Overexposed: {overexposed*100:.1f}% "
                f"pixels clipped at white"
            )
```

> **💡 What is a histogram and what is "clipping"?**
>
> A **histogram** counts how many pixels have each brightness value (0–255):
>
> ```
> Frequency
> │
> │    ┌──┐
> │    │  │   ┌──┐
> │ ┌──┤  │   │  │  ┌──┐
> │ │  │  │   │  │  │  │ ┌──┐
> │ │  │  │   │  ├──┤  │ │  │
> └─┴──┴──┴───┴──┴──┴──┴─┴──┴──
>   0    50   100  150  200  255
>         Pixel Brightness →
> ```
>
> **Clipping** happens when the camera sensor can't handle the brightness range:
>
> | Problem | What Happens | In the Histogram |
> |---------|-------------|------------------|
> | **Underexposed** (too dark) | Dark areas become pure black (0). Detail is lost forever. | Huge spike at bin 0 |
> | **Overexposed** (too bright) | Bright areas become pure white (255). Detail is lost forever. | Huge spike at bin 255 |
>
> The threshold `0.1` means: if more than **10%** of all pixels are clipped to pure black (0) or pure white (255), the image has a serious exposure problem.
>
> **Why does this matter for defects?**  
> If a scratch on a bright surface is overexposed to pure white, it's invisible — the model can't learn to detect it. Similarly, defects in shadows that clip to black are lost.

---

#### Returning the Result

```python
        return {                                     # Line 101: Return the validation result
            'path': image_path,                      # Line 102: Which image was checked
            'passed': len(issues) == 0,              # Line 103: Passed if no issues found
            'issues': issues,                        # Line 104: List of problems (empty if passed)
            'metrics': metrics                       # Line 105: All measured values
        }
```

> **💡 The result dictionary structure:**
>
> ```python
> # Example PASSED result:
> {
>     'path': 'dataset/raw/good/good_20260216_143021.png',
>     'passed': True,
>     'issues': [],                                  # Empty = no problems
>     'metrics': {
>         'resolution': (1280, 960),
>         'brightness': 127.35,
>         'contrast': 45.82,
>         'sharpness': 523.17,
>         'underexposed_pct': 0.02,
>         'overexposed_pct': 0.01
>     }
> }
>
> # Example FAILED result:
> {
>     'path': 'dataset/raw/scratch/scratch_20260216_143045.png',
>     'passed': False,
>     'issues': [
>         'Blurry: sharpness=42.3 (min: 100.0)',
>         'Too dark: brightness=35.2 (min: 50)'
>     ],
>     'metrics': {
>         'resolution': (640, 480),
>         'brightness': 35.21,
>         'contrast': 28.44,
>         'sharpness': 42.33,
>         'underexposed_pct': 3.21,
>         'overexposed_pct': 0.0
>     }
> }
> ```

---

### Validating a Directory — `validate_directory`

This method runs `validate()` on **every image** in a directory:

```python
    def validate_directory(self, directory: str,     # Line 108
                           extensions: List[str]      # Line 109: Which file types to check
                               = ['png', 'jpg', 'jpeg']
                           ) -> Dict:
        """                                          # Lines 110-118: Docstring
        Validate all images in a directory

        Args:
            directory: Path to directory with images
            extensions: List of file extensions to check

        Returns:
            Summary dict with pass/fail counts and problem images
        """
        results = {                                  # Line 120: Initialize summary
            'total': 0,                              # Line 121: Total images checked
            'passed': 0,                             # Line 122: How many passed
            'failed': 0,                             # Line 123: How many failed
            'problem_images': []                     # Line 124: Details of failed images
        }
```

```python
        # Find all images
        patterns = [                                 # Line 128: Build glob patterns
            f"{directory}/**/*.{ext}"                #   e.g., "dataset/raw/**/*.png"
            for ext in extensions                    #         "dataset/raw/**/*.jpg"
        ]                                            #         "dataset/raw/**/*.jpeg"

        image_paths = []                             # Line 129: Collect all found paths
        for pattern in patterns:                     # Line 130: For each pattern...
            image_paths.extend(                      # Line 131: Add all matches to the list
                glob.glob(pattern, recursive=True)   #   recursive=True → search subdirectories too
            )

        print(f"Found {len(image_paths)} images "
              f"to validate...")                     # Line 133: Tell user how many images found
```

> **💡 How `glob.glob()` finds files:**
>
> ```python
> # Pattern: "dataset/raw/**/*.png"
> #
> #  dataset/raw/  = starting directory
> #  **/           = any depth of subdirectories (0, 1, 2, ...)
> #  *.png         = any filename ending in .png
> #
> # This finds:
> #   dataset/raw/good/good_001.png           ✓
> #   dataset/raw/scratch/scratch_001.png     ✓
> #   dataset/raw/subfolder/deep/image.png    ✓
> #   dataset/raw/photo.jpg                   ✗ (wrong extension)
> ```
>
> **Why 3 extensions?**  
> Cameras and phones save in different formats. PNG is lossless (our `camera_capture.py` uses it), but Roboflow datasets often come as JPG. By checking all three, we validate any image source.

```python
        for img_path in image_paths:                 # Line 135: Loop through ALL images
            result = self.validate(img_path)          # Line 136: Check each one
            results['total'] += 1                    # Line 137: Increment total count

            if result['passed']:                     # Line 139: Did it pass?
                results['passed'] += 1               # Line 140: Count passed
            else:
                results['failed'] += 1               # Line 142: Count failed
                results['problem_images'].append(    # Line 143: Save failed details
                    result
                )

        return results                               # Line 145: Return summary
```

> **💡 The validation flow for a directory:**
>
> ```
> validate_directory("dataset/raw")
>        │
>        ▼
> Find all *.png, *.jpg, *.jpeg
>        │
>        ▼  (loops through each)
> ┌──────────────────────────────────────────┐
> │ Image 1: good_001.png → validate()       │
> │   Resolution: ✓  Brightness: ✓           │
> │   Contrast: ✓  Sharpness: ✓  Exposure: ✓ │
> │   → PASSED                                │
> ├──────────────────────────────────────────┤
> │ Image 2: scratch_003.png → validate()    │
> │   Resolution: ✓  Brightness: ✓           │
> │   Contrast: ✓  Sharpness: ✗  Exposure: ✓ │
> │   → FAILED (blurry)                       │
> ├──────────────────────────────────────────┤
> │ Image 3: crack_001.png → validate()      │
> │   Resolution: ✓  Brightness: ✗           │
> │   Contrast: ✓  Sharpness: ✓  Exposure: ✗ │
> │   → FAILED (too dark, underexposed)       │
> └──────────────────────────────────────────┘
>        │
>        ▼
> Return: { total: 3, passed: 1, failed: 2,
>           problem_images: [...] }
> ```

---

### Printing the Report — `print_report`

```python
    def print_report(self, directory: str):           # Line 147
        """Print a formatted validation report        # Line 148
        for a directory"""
        results = self.validate_directory(directory)  # Line 149: Run all validations

        print("\n" + "="*60)                          # Line 151: Header divider
        print("IMAGE QUALITY VALIDATION REPORT")      # Line 152: Title
        print("="*60)                                 # Line 153: Header divider
        print(f"Directory: {directory}")               # Line 154: Which directory was checked
        print(f"Total images: {results['total']}")     # Line 155: Count

        print(f"Passed: {results['passed']} "          # Line 156: Passed count + percentage
              f"({results['passed']/"
              f"max(1,results['total'])*100:.1f}%)")

        print(f"Failed: {results['failed']} "          # Line 157: Failed count + percentage
              f"({results['failed']/"
              f"max(1,results['total'])*100:.1f}%)")
```

> **💡 Why `max(1, results['total'])`?**
>
> This prevents **division by zero**. If no images were found (`total = 0`), dividing by zero would crash the program. `max(1, 0)` returns `1`, giving `0/1 = 0%` instead of a crash. This is a common defensive programming trick.

```python
        if results['problem_images']:                # Line 159: Any problems?
            print("\n" + "-"*60)                      # Line 160: Subheader
            print("PROBLEM IMAGES:")                  # Line 161: Section title
            print("-"*60)                             # Line 162

            for prob in results['problem_images']     # Line 163: Show first 10 problems
                                        [:10]:
                print(f"\n  {Path(prob['path'])       # Line 164: Print filename only
                                    .name}")          #   Path.name = just "image.png"
                                                      #   (not the full path)
                for issue in prob['issues']:          # Line 165: Print each issue
                    print(f"    ⚠ {issue}")            # Line 166: With warning symbol

            if len(results['problem_images']) > 10:   # Line 168: More than 10 problems?
                remaining = len(results               # Line 169
                    ['problem_images']) - 10
                print(f"\n  ... and {remaining} "
                      f"more problem images")

        print("\n" + "="*60)                          # Line 171: Footer
        return results                               # Line 172: Return results (for programmatic use)
```

> **💡 Why limit to 10?**
>
> If you're validating 500 images and 200 fail, printing all 200 would flood the terminal. Showing the first 10 gives you enough to diagnose the pattern (e.g., "all dark" → lighting problem, "all blurry" → focus problem) without overwhelming you.

> **💡 Example terminal output:**
>
> ```
> Found 87 images to validate...
>
> ============================================================
> IMAGE QUALITY VALIDATION REPORT
> ============================================================
> Directory: dataset/raw
> Total images: 87
> Passed: 79 (90.8%)
> Failed: 8 (9.2%)
>
> ------------------------------------------------------------
> PROBLEM IMAGES:
> ------------------------------------------------------------
>
>   scratch_20260216_143045_789012.png
>     ⚠ Blurry: sharpness=42.3 (min: 100.0)
>
>   crack_20260216_143102_345678.png
>     ⚠ Too dark: brightness=35.2 (min: 50)
>     ⚠ Underexposed: 12.3% pixels clipped at black
>
>   good_20260216_150000_111111.png
>     ⚠ Low contrast: std=15.2 (min: 20.0)
>
> ============================================================
> ```

---

## Main Function — `main()`

This is the command-line interface that lets you run the script from the terminal:

```python
def main():                                          # Line 175
    """Run quality validation on dataset directory""" # Line 176

    import argparse                                  # Line 177: Import argument parser
                                                     #   (imported here, not at top, because
                                                     #   it's only needed for CLI usage)

    parser = argparse.ArgumentParser(                # Line 179: Create parser
        description='Validate image quality '
                    'for training'
    )

    parser.add_argument(                             # Line 180: First argument (positional)
        'directory',                                 #   Name: "directory"
        nargs='?',                                   #   nargs='?' → optional positional arg
        default='dataset/raw',                       #   Default if not provided
        help='Directory containing images to validate'
    )

    parser.add_argument(                             # Line 182: --min-width flag
        '--min-width',
        type=int,                                    #   Must be an integer
        default=640,                                 #   Default: 640 pixels
        help='Minimum image width'
    )

    parser.add_argument(                             # Line 184: --min-height flag
        '--min-height',
        type=int,
        default=480,
        help='Minimum image height'
    )

    parser.add_argument(                             # Line 186: --min-sharpness flag
        '--min-sharpness',
        type=float,                                  #   Must be a float (decimal number)
        default=100.0,
        help='Minimum sharpness threshold'
    )

    args = parser.parse_args()                       # Line 189: Parse command-line arguments
```

> **💡 What is `argparse`?**
>
> `argparse` is Python's built-in library for parsing command-line arguments. It lets you write:
> ```bash
> python validate_images.py dataset/raw --min-width 800 --min-sharpness 50
> ```
>
> | Argument | Type | What It Sets |
> |----------|------|-------------|
> | `dataset/raw` | Positional (optional) | `args.directory` → which folder to check |
> | `--min-width 800` | Named flag | `args.min_width` → override minimum width |
> | `--min-height 600` | Named flag | `args.min_height` → override minimum height |
> | `--min-sharpness 50` | Named flag | `args.min_sharpness` → override blur threshold |
>
> **Why `nargs='?'`?**  
> This makes the positional argument optional. Without it, you'd HAVE to provide a directory every time. With `nargs='?'`, you can just run `python validate_images.py` and it defaults to `dataset/raw`.
>
> **Why import `argparse` inside the function?**  
> This is a micro-optimization. When other scripts import `ImageQualityValidator` from this file, they don't need `argparse`. Importing it inside `main()` means it's only loaded when running the script directly from the command line.

```python
    validator = ImageQualityValidator(               # Line 191: Create the validator
        min_resolution=(                             # Line 192: Pass custom resolution
            args.min_width,
            args.min_height
        ),
        min_sharpness=args.min_sharpness             # Line 193: Pass custom sharpness
    )

    validator.print_report(args.directory)            # Line 196: Run and print the report
```

> **💡 Note:** Only `min_resolution` and `min_sharpness` are configurable from the command line. `brightness_range` and `min_contrast` use the class defaults (`(50, 205)` and `20.0`). You could add `--min-brightness` and `--max-brightness` flags if needed.

---

## Script Entry Point

```python
if __name__ == "__main__":                           # Line 199
    main()                                           # Line 200: Run the CLI
```

> **💡 What does `if __name__ == "__main__":` mean?**
>
> - When you run `python validate_images.py` directly → `__name__` is `"__main__"` → `main()` runs
> - When you do `from validate_images import ImageQualityValidator` → `__name__` is `"validate_images"` → `main()` is **skipped**
>
> This lets the file work as both:
> - A **standalone CLI tool**: `python validate_images.py dataset/raw`
> - An **importable module**: `run_pipeline.py` imports `ImageQualityValidator` directly

---

## How to Run

```bash
# Basic usage — validate images in dataset/raw (default directory)
python scripts/validate_images.py

# Validate a specific directory
python scripts/validate_images.py dataset/raw

# With custom thresholds
python scripts/validate_images.py dataset/raw --min-width 800 --min-height 600 --min-sharpness 50

# From within the pipeline
python scripts/run_pipeline.py --mode full    # Automatically runs validation first
```

---

## How It Connects to Other Scripts

```
camera_capture.py
       │
       │  Saves images to dataset/raw/{label}/
       │
       ▼
validate_images.py  ◄──── YOU ARE HERE
       │
       │  Checks quality of captured images:
       │  ✓ Resolution ≥ 640×480
       │  ✓ Brightness in range 50–205
       │  ✓ Contrast (std dev) ≥ 20
       │  ✓ Sharpness (Laplacian) ≥ 100
       │  ✓ Exposure clipping < 10%
       │
       │  If images pass → continue pipeline
       │  If images fail → fix lighting/focus/camera settings
       │
       ▼
prepare_dataset.py / split_dataset.py
       │
       │  Organizes validated images into train/val/test
       │
       ▼
train_model.py
       │
       │  Trains YOLO on the validated, split dataset
       ▼
```

> **Who imports from this file?**
>
> | Script | How It Uses `validate_images.py` |
> |--------|------|
> | `run_pipeline.py` | Imports `ImageQualityValidator` to run validation as Step 1 of the full pipeline |
> | **CLI (you)** | Run directly to check image quality before training |
>
> **The output of `camera_capture.py` (`dataset/raw/`) is the input for `validate_images.py`.**  
> **The output of `validate_images.py` (pass/fail report) tells you whether to proceed with training or fix your data first.**

---

*This document covers every line of `validate_images.py`. For other scripts, see the corresponding guides in this folder.*

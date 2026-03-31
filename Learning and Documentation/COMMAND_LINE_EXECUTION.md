# Command Line Execution Guide

This document lists the command line execution instructions for the entire defect detection pipeline, from dataset capture to real-time inference.

## 1. Environment Setup

Before running any scripts, ensure your virtual environment is activated and you are in the project root directory.

```bash
# Activate environment (Windows)
.\defect_env_gpu\Scripts\activate
# OR 
.\defect_env_gpu311\Scripts\activate

# Most commands should be run from the project root directory:
# python scripts/script_name.py
```

## 2. Dataset Collection & Preparation

### 2.1 Camera Capture (`camera_capture.py`)
Interactive tool to capture images from the Basler camera. It prompts for a save mode (original/greyscale) up-front and requires keyboard inputs (`g`/`s`/`c`/`d`/`r`/`y`) to classify and save the image.

```bash
# Run in interactive mode
python scripts/camera_capture.py
```

### 2.2 Prepare Dataset (`prepare_dataset.py`)
Merges raw images and CVAT annotations into a unified raw folder. Configuration paths are hardcoded in the file.

```bash
# Execute dataset merge
python scripts/prepare_dataset.py
```

### 2.3 Split Dataset (`split_dataset.py`)
Splits the raw dataset into training, validation, and test sets.

```bash
# Default split (70% train, 15% val, 15% test)
python scripts/split_dataset.py 

# Custom split using specific ratios and output directory
python scripts/split_dataset.py dataset/raw --output dataset/split --train 0.8 --val 0.1 --test 0.1

# Move files instead of copying to save disk space
python scripts/split_dataset.py --move
```

### 2.4 Augment Dataset (`augment_dataset.py`)
Applies augmentations to the dataset to improve model robustness, particularly for minority classes.

```bash
# Default augmentation (creates 3 copies per image, medium intensity)
python scripts/augment_dataset.py

# Heavy augmentation with 5 copies per image
python scripts/augment_dataset.py --level heavy --multiplier 5

# Preview augmentations visually in a window without saving
python scripts/augment_dataset.py --preview
```

### 2.5 Greyscale Converter (`greyscale_converter.py`)
Utility to convert images to a rust-aware greyscale format (CLAHE enhanced). This feature is also integrated directly into `camera_capture.py`.

```bash
# Convert a specific image or an entire folder
python scripts/greyscale_converter.py --input path/to/input --output path/to/output

# Adjust the darkening strength of the rust mask (0.0 to 1.0)
python scripts/greyscale_converter.py --input path/to/input --strength 0.65
```

## 3. Model Training & Evaluation

### 3.1 Train Model (`train_model.py`)
Trains the YOLO model using the prepared dataset and hyperparameter presets.

```bash
# Standard training using the default parameter preset (good_vs_rust_optimized)
python scripts/train_model.py

# Resume training from the last interrupted checkpoint
python scripts/train_model.py --resume

# Override specific training parameters on the fly
python scripts/train_model.py --epochs 100 --batch 32 --imgsz 640

# Disable Weights & Biases (wandb) logging entirely
python scripts/train_model.py --no-wandb
```

### 3.2 Evaluate Model (`evaluate_model.py`)
Evaluates a trained YOLO model against the test or validation dataset to calculate precision, recall, mAP, and F1-scores.

```bash
# Evaluate the best model on the test set (default)
python scripts/evaluate_model.py --model models/best.pt

# Evaluate on the validation set with custom thresholds
python scripts/evaluate_model.py --model models/best.pt --split val --conf 0.25 --iou 0.6
```

### 3.3 Compare Runs (`compare_runs.py`)
Scans multiple training runs and generates visual plots and metrics to help determine the best model.

```bash
# Compare runs in the default directories
python scripts/compare_runs.py

# Show detailed per-run information
python scripts/compare_runs.py --detailed
```

## 4. Inference & Real-time Detection

### 4.1 Defect Detector (`defect_detector.py`)
Runs the final inference engine on live camera feeds, video files, or individual images.

```bash
# 1. Live Camera Feed Mode (defaults to Basler camera)
python scripts/defect_detector.py --mode live --model models/best.pt

# 2. Image Mode (requires an input source)
python scripts/defect_detector.py --mode image --model models/best.pt --source path/to/image.jpg

# 3. Video Mode (requires an input source)
python scripts/defect_detector.py --mode video --model models/best.pt --source path/to/video.mp4

# Extra Options:
# Use mock camera for local testing without physical hardware
python scripts/defect_detector.py --mode live --mock

# Change the defect classification confidence threshold
python scripts/defect_detector.py --mode image --source my_image.jpg --conf 0.5
```

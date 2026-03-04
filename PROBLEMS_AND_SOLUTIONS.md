# 🔧 Problems Faced & Solutions Found

> **AI Agent Instructions**: This is a **living document**. Every time a problem is encountered and resolved during development, **add a new entry** at the top of the relevant category.  
> Use the template at the bottom of this file for consistency. Keep entries concise but include enough detail to be useful for learning and debugging similar issues in the future.

---

## 📊 Summary

| Category | Total Issues | Resolved | Open |
|----------|-------------|----------|------|
| Environment & Setup | 4 | 4 | 0 |
| Training & Model | 3 | 3 | 0 |
| Data & Annotation | 2 | 2 | 0 |
| Deployment & Infra | 1 | 1 | 0 |

**Last Updated**: `2026-03-04`

---

## 🖥️ Environment & Setup

### P-001: GPU Not Detected for Training
- **Date**: 2026-02-16
- **Severity**: 🔴 High
- **Status**: ✅ Resolved

**Problem**:  
Training was running on CPU despite having a GPU available. The `defect_env` virtual environment had CPU-only PyTorch installed. Training was extremely slow (~10x slower than expected).

**Root Cause**:  
The default `pip install torch` installs the CPU-only version. CUDA-compatible PyTorch requires a specific install command pointing to the correct CUDA wheel index.

**Solution**:  
Created a separate GPU-enabled virtual environment (`defect_env_gpu`) with CUDA-compatible PyTorch:
```bash
python -m venv defect_env_gpu
defect_env_gpu\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Lesson Learned**:  
Always verify GPU availability before starting training with:
```python
import torch
print(torch.cuda.is_available())       # Should be True
print(torch.cuda.get_device_name(0))   # Should show your GPU
```

---

### P-002: Two Separate Virtual Environments Needed
- **Date**: 2026-02-16
- **Severity**: 🟡 Medium
- **Status**: ✅ Resolved

**Problem**:  
Confusion about which virtual environment to activate (`defect_env` vs `defect_env_gpu`). Different team members were accidentally using the CPU environment for training.

**Root Cause**:  
CPU and GPU versions of PyTorch conflict — can't have both in the same environment. No documentation existed about which env to use when.

**Solution**:  
- Documented both environments in `PROJECT_DOCS.md` Setup section
- Added environment check in `train_model.py` → `check_environment()` prints GPU status at the start of every training run
- Convention: use `defect_env_gpu` for training, `defect_env` only for non-GPU tasks

**Lesson Learned**:  
Always add an environment auto-check at the start of GPU-dependent scripts.

---

### P-003: Basler Pylon SDK Not Found (pypylon ImportError)
- **Date**: 2026-02-16
- **Severity**: 🟡 Medium
- **Status**: ✅ Resolved

**Problem**:  
`ImportError: No module named 'pypylon'` even after `pip install pypylon`. The camera capture script couldn't connect to the Basler camera.

**Root Cause**:  
`pypylon` is a Python wrapper but requires the **Basler Pylon SDK** to be installed separately on the system. The SDK provides the actual camera drivers.

**Solution**:  
1. Download and install Pylon SDK from [Basler website](https://www.baslerweb.com/en/downloads/software-downloads/)
2. Then `pip install pypylon>=3.0.0`
3. Added `MockCamera` fallback class in `camera_capture.py` so the script works even without a physical camera connected

**Lesson Learned**:  
Hardware SDK dependencies can't be resolved with `pip` alone — always document system-level prerequisites.

---

### P-004: CVAT Annotation Tool Docker Setup Issues
- **Date**: 2026-02-16
- **Severity**: 🟡 Medium
- **Status**: ✅ Resolved

**Problem**:  
CVAT annotation server failed to start. Docker containers weren't running correctly, and the admin account creation step was unclear.

**Root Cause**:  
- Docker Desktop needed to be running before starting CVAT
- The admin account needs to be created via a `docker exec` command after containers are up
- Port conflicts on `localhost:8080`

**Solution**:  
1. Created `start_cvat.bat` and `stop_cvat.bat` scripts for easy server management
2. Documented the full CVAT setup flow:
   ```bash
   # Start CVAT
   start_cvat.bat
   # Wait for containers to fully start (~30 seconds)
   # Create admin account
   docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
   # Access at http://localhost:8080
   ```
3. Exported annotations in **YOLO 1.1 format** from CVAT

**Lesson Learned**:  
Create wrapper scripts (`start_cvat.bat`, `stop_cvat.bat`) for complex multi-step tool setups. Document the full flow end-to-end.

---

## 🧠 Training & Model

### P-005: Training Stopped Early — Low Patience with Small Dataset
- **Date**: 2026-02-16
- **Severity**: 🔴 High
- **Status**: ✅ Resolved

**Problem**:  
Model training stopped after only ~30 epochs (out of 100) due to early stopping. The model hadn't converged yet, and the final mAP was poor.

**Root Cause**:  
- Dataset was very small (~130 images total, ~87 training images across 3 classes)
- `patience=20` was too aggressive for a small dataset — the model needs more time to learn from limited data
- Validation metrics were noisy with so few validation images (only 19), triggering early stopping prematurely

**Solution**:  
1. Increased `patience` from 20 to 50 in `train_model.py`
2. Created `config/hyperparams.yaml` with a `small_dataset` preset: `epochs=300`, `patience=50`, `lr0=0.001`, `optimizer=AdamW`
3. Added heavier augmentation parameters to compensate for limited data
4. Created `scripts/augment_dataset.py` for offline data augmentation (5x multiplier)

**Lesson Learned**:  
For small datasets (<200 images): use lower learning rate, higher patience, more augmentation, and more epochs. The `small_dataset` preset in `config/hyperparams.yaml` encodes these learnings.

---

### P-006: Understanding YOLO Augmentation Engine Internals
- **Date**: 2026-02-27
- **Severity**: 🟢 Low
- **Status**: ✅ Resolved

**Problem**:  
Needed to explain to the manager how Ultralytics handles augmentations internally (Mosaic, MixUp, CutMix). The documentation wasn't clear on what happens under the hood.

**Root Cause**:  
Ultralytics augmentations are applied dynamically during training by the data loader. They're not visible in the dataset — they happen on-the-fly at each batch.

**Solution**:  
Researched the Ultralytics engine source code and documented:
- **Mosaic**: Combines 4 images into one, helping the model see objects at different scales and contexts
- **MixUp**: Blends two images together with a random ratio, acting as regularization
- **CutMix**: Cuts a patch from one image and pastes it onto another
- All are controlled via training parameters (`mosaic=1.0`, `mixup=0.1`, etc.)
- Created `Understanding_YOLO_Complete_Guide.docx` as a reference document

**Lesson Learned**:  
Online augmentations (applied during training) complement offline augmentations (applied beforehand). Use both for maximum data diversity.

---

### P-007: No Way to Compare Training Experiments
- **Date**: 2026-03-04
- **Severity**: 🟡 Medium
- **Status**: ✅ Resolved

**Problem**:  
After multiple training runs with different hyperparameters, there was no easy way to compare which run performed best. Had to manually open each `results.csv` file.

**Root Cause**:  
No metrics tracking or comparison tooling existed in the project.

**Solution**:  
Created `scripts/compare_runs.py` which:
- Auto-discovers all training runs in `runs/detect/` and `models/`
- Parses `results.csv` from each run
- Prints a comparison table (mAP50, mAP50-95, Precision, Recall)
- Generates overlay plots and JSON summary

**Lesson Learned**:  
Always build experiment tracking into your ML pipeline from the start. Even a simple CSV comparison script saves hours of manual work.

---

## 📁 Data & Annotation

### P-008: CVAT Exports Didn't Match Image Directories
- **Date**: 2026-02-16
- **Severity**: 🔴 High
- **Status**: ✅ Resolved

**Problem**:  
After exporting annotations from CVAT in YOLO format, the label `.txt` files were in nested directories that didn't match the image file locations. The training script couldn't pair images with their labels.

**Root Cause**:  
CVAT organizes exports by task, with labels in `obj_Train_data/<subfolder>/` paths. Images were stored in completely different directories (`Good/`, `Automation/`, `Classroom/`).

**Solution**:  
Created `scripts/prepare_dataset.py` which:
1. Maps each CVAT export to its corresponding image directory (3 annotation sources)
2. Merges matching image-label pairs into `dataset/raw/`
3. Splits into `dataset/images/{train,val,test}` and `dataset/labels/{train,val,test}`
4. Reports class distribution statistics after merging

**Lesson Learned**:  
Always build a dataset preparation script that handles the gap between annotation tool exports and training format. Don't rely on manual file copying — it's error-prone.

---

### P-009: Class Labels Changed Mid-Project
- **Date**: 2026-02-27
- **Severity**: 🟡 Medium
- **Status**: ✅ Resolved

**Problem**:  
Originally started with classes `scratch`, `crack`, `dent`, `discoloration` but later changed to `Good`, `Flat_line`, `Unwash` based on actual production defect types. Old configs and references were inconsistent.

**Root Cause**:  
Requirements evolved as the project moved from a generic defect detection concept to actual production defects. Multiple config files and scripts referenced the old class names.

**Solution**:  
1. Updated `config/dataset.yaml` with new classes: `Good` (0), `Flat_line` (1), `Unwash` (2)
2. Updated `config/system_config.json` classes array
3. Updated `camera_capture.py` capture key mappings
4. Re-annotated images with correct labels in CVAT
5. Documented class definitions in `PROJECT_DOCS.md` Detection Classes table

**Lesson Learned**:  
Define class names early and keep a **single source of truth** (`config/dataset.yaml`). When classes change, use a checklist to update all dependent files.

---

## 🚀 Deployment & Infrastructure

### P-010: Dashboard URL Not Loading
- **Date**: 2026-02-13
- **Severity**: 🟡 Medium
- **Status**: ✅ Resolved

**Problem**:  
The WH_std dashboard URL was not loading. The embedded dashboard page showed a blank screen.

**Root Cause**:  
Configuration mismatch in `config.js` — the URL entry in `dashboard_urls.csv` was either missing or incorrectly formatted.

**Solution**:  
Debugged by checking `config.js`, `dashboard_urls.csv`, and `app.js` for the URL mapping. Fixed the configuration entry and verified the dashboard loaded correctly.

**Lesson Learned**:  
When adding new dashboard URLs, always verify the entry exists in both the CSV config and the JavaScript routing logic.

---

## 📝 Template for New Entries

> **Copy this template when adding a new problem:**

```markdown
### P-XXX: [Short Title]
- **Date**: YYYY-MM-DD
- **Severity**: 🔴 High / 🟡 Medium / 🟢 Low
- **Status**: ✅ Resolved / 🔄 In Progress / ❌ Open

**Problem**:  
[What went wrong? What was the symptom?]

**Root Cause**:  
[Why did it happen?]

**Solution**:  
[How was it fixed? Include code/commands if applicable.]

**Lesson Learned**:  
[What should you do differently next time?]
```

---

*This document is automatically updated whenever a problem is encountered and resolved. Each entry includes the context, root cause, fix, and lesson learned for future reference.*

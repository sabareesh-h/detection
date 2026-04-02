"""
Fix NaN Training Issues
=======================
Diagnoses and fixes two common causes of NaN loss in YOLO training:
1. Empty/corrupt label files (0-byte .txt files)
2. Mismatched class counts between dataset.yaml and actual training data
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "scripts" / "dataset"

SPLITS = ["train", "val", "test"]


def check_empty_labels(split="train", fix=False):
    """Find (and optionally remove) empty label files."""
    label_dir = DATASET_ROOT / "labels" / split
    image_dir = DATASET_ROOT / "images" / split

    if not label_dir.exists():
        print(f"  [SKIP] Label dir not found: {label_dir}")
        return []

    empty_files = []
    for txt in label_dir.glob("*.txt"):
        if txt.stat().st_size == 0:
            empty_files.append(txt)

    if not empty_files:
        print(f"  [OK] No empty label files in '{split}' split.")
        return []

    print(f"\n  [WARNING] Found {len(empty_files)} empty label file(s) in '{split}':")
    for f in empty_files:
        print(f"    - {f.name}")
        # Also check if corresponding image exists
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            img = image_dir / (f.stem + ext)
            if img.exists():
                print(f"      └─ Image exists: {img.name}")

    if fix:
        print(f"\n  Removing {len(empty_files)} empty label file(s) and their images...")
        removed_labels = 0
        removed_images = 0
        for txt in empty_files:
            txt.unlink()
            removed_labels += 1
            # Remove matching image too (no label = unusable for detection training)
            for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                img = image_dir / (txt.stem + ext)
                if img.exists():
                    img.unlink()
                    removed_images += 1
        print(f"  Removed {removed_labels} label files and {removed_images} images.")
    else:
        print("\n  Run with fix=True to remove them.")

    return empty_files


def check_class_distribution(split="train"):
    """Show which class IDs are actually used in label files."""
    label_dir = DATASET_ROOT / "labels" / split
    if not label_dir.exists():
        print(f"  [SKIP] Label dir not found: {label_dir}")
        return {}

    class_counts = defaultdict(int)
    total_labels = 0
    files_with_labels = 0
    files_empty = 0

    for txt in label_dir.glob("*.txt"):
        lines = txt.read_text().strip().splitlines()
        if not lines:
            files_empty += 1
            continue
        files_with_labels += 1
        for line in lines:
            try:
                cls_id = int(line.split()[0])
                class_counts[cls_id] += 1
                total_labels += 1
            except (ValueError, IndexError):
                print(f"  [BAD LINE] {txt.name}: '{line}'")

    print(f"\n  Class distribution in '{split}' split:")
    print(f"  Files with labels: {files_with_labels}")
    print(f"  Empty label files: {files_empty}")
    print(f"  Total annotations: {total_labels}")
    if class_counts:
        print(f"  Class ID → count:")
        for cls_id in sorted(class_counts):
            print(f"    Class {cls_id:2d}: {class_counts[cls_id]:5d} annotations")
        present = set(class_counts.keys())
        print(f"  Classes present: {sorted(present)}")
    else:
        print("  No annotations found!")
    return class_counts


def suggest_dataset_yaml_fix(train_classes, val_classes=None):
    """Suggest a corrected dataset.yaml based on actual classes."""
    all_classes = set(train_classes.keys())
    if val_classes:
        all_classes |= set(val_classes.keys())

    CLASS_NAMES = {
        0: "Good(Top)",
        1: "Rust(Top)",
        2: "Rust(Mid)",
        3: "Rust(Bottom)",
        4: "Rust(Thread)",
        5: "Good(Mid)",
        6: "Good(Thread)",
        7: "Good(Bottom)",
        8: "Scratch(Top)",
        9: "Scratch(Mid)",
        10: "Scratch(Thread)",
        11: "Scratch(Bottom)",
    }

    print("\n" + "=" * 60)
    print("SUGGESTION: Update dataset.yaml to only include present classes")
    print("=" * 60)
    if not all_classes:
        print("No classes found in dataset — check your labels directory!")
        return

    # Check if we can remap to contiguous IDs
    max_id = max(all_classes)
    missing = [i for i in range(max_id + 1) if i not in all_classes]

    if missing:
        print(f"\n  WARNING: Class IDs are NOT contiguous.")
        print(f"  Present IDs: {sorted(all_classes)}")
        print(f"  Missing IDs: {missing}")
        print(f"\n  YOLO requires contiguous class IDs 0..N-1.")
        print(f"  You need to either:")
        print(f"    A) Add images for missing classes {[CLASS_NAMES.get(m, f'class_{m}') for m in missing]}")
        print(f"    B) Re-label your data with new contiguous IDs")
        print(f"       Remapping suggestion:")
        new_id = 0
        remap = {}
        for old_id in sorted(all_classes):
            remap[old_id] = new_id
            print(f"       Class {old_id} ({CLASS_NAMES.get(old_id, '?')}) → new ID {new_id}")
            new_id += 1
    else:
        print(f"\n  Class IDs are contiguous (0..{max_id}).")
        print(f"  Suggested dataset.yaml names section:")
        print(f"  names:")
        for cls_id in sorted(all_classes):
            print(f"    {cls_id}: {CLASS_NAMES.get(cls_id, f'class_{cls_id}')}")
        print(f"  nc: {len(all_classes)}")


def main():
    print("=" * 60)
    print("NaN TRAINING DIAGNOSTIC")
    print("=" * 60)
    print(f"Dataset root: {DATASET_ROOT}")

    # --- Step 1: Check for empty labels ---
    print("\n[1] CHECKING FOR EMPTY LABEL FILES")
    print("-" * 40)

    fix_mode = "--fix" in sys.argv

    all_empty = []
    for split in SPLITS:
        print(f"\n  Split: {split}")
        empty = check_empty_labels(split=split, fix=fix_mode)
        all_empty.extend(empty)

    # --- Step 2: Class distribution ---
    print("\n\n[2] CLASS DISTRIBUTION CHECK")
    print("-" * 40)

    train_classes = check_class_distribution("train")
    val_classes = check_class_distribution("val")

    # --- Step 3: Suggest fix ---
    suggest_dataset_yaml_fix(train_classes, val_classes)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY & NEXT STEPS")
    print("=" * 60)

    if all_empty:
        if fix_mode:
            print(f"[FIXED] Removed {len(all_empty)} empty label files (and their images).")
        else:
            print(f"[ACTION NEEDED] {len(all_empty)} empty label files found.")
            print(f"  → Re-run with --fix to remove them:")
            print(f"     python scripts/fix_nan_training.py --fix")

    missing_classes = set()
    for cls_id in range(12):  # dataset.yaml has 12 classes
        if cls_id not in train_classes and cls_id not in val_classes:
            missing_classes.add(cls_id)

    CLASS_NAMES = {0:"Good(Top)",1:"Rust(Top)",2:"Rust(Mid)",3:"Rust(Bottom)",
                   4:"Rust(Thread)",5:"Good(Mid)",6:"Good(Thread)",7:"Good(Bottom)",
                   8:"Scratch(Top)",9:"Scratch(Mid)",10:"Scratch(Thread)",11:"Scratch(Bottom)"}

    if missing_classes:
        print(f"\n[ACTION NEEDED] {len(missing_classes)} class(es) declared in dataset.yaml but have ZERO training images:")
        for cls_id in sorted(missing_classes):
            print(f"  Class {cls_id}: {CLASS_NAMES.get(cls_id, '?')}")
        print(f"\n  This causes NaN loss. Fix by:")
        print(f"  Option A: Add training images for these classes.")
        print(f"  Option B: Update dataset.yaml nc and names to only include present classes.")
        print(f"            Then re-label any images that use old class IDs.")

    print("\nAfter fixing, re-run training with a FRESH start (not --resume).")
    print("Recommended: use smaller img_size=640 or batch_size=4 for first test run.")


if __name__ == "__main__":
    main()

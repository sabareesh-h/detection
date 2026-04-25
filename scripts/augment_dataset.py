"""
=============================================================
  augment_dataset.py  --  Defect detection pipeline script
=============================================================
HOW TO USE
----------
python augment_dataset.py [-h] [--input INPUT] [--labels LABELS]

Examples:
# Augment training set with 2x copies (medium intensity)
  python augment_dataset.py --input dataset/images/train --labels dataset/labels/train --multiplier 2

  # Heavy augmentation to separate output directory
  python augment_dataset.py --input dataset/images/train --labels dataset/labels/train \
      --output-images dataset/images/train_aug --output-labels dataset/labels/train_aug \
      --level heavy --multiplier 10

  # Preview augmentations without saving
  python augment_dataset.py --input dataset/images/train --labels dataset/labels/train --preview

FLAGS
-----
-h, --help            show this help message and exit
    --input INPUT         Input images directory (default: dataset/images/train)
    --labels LABELS       Input labels directory (default: dataset/labels/train)
    --output-images OUTPUT_IMAGES
    Output images directory (default: same as input â€” in-
    place)
    --output-labels OUTPUT_LABELS
    Output labels directory (default: same as labels â€” in-
    place)
    --multiplier MULTIPLIER
    Number of augmented copies per image (default: 2)
    --level {light,medium,heavy}
    Augmentation intensity (default: medium)
    --seed SEED           Random seed for reproducibility
    --preview             Preview augmentations in a window (no files saved)
    --no-copy-originals   Do not copy original images to output dir
    Examples:
    # Augment training set with 2x copies (medium intensity)
    python augment_dataset.py --input dataset/images/train --labels dataset/labels/train --multiplier 2
    # Heavy augmentation to separate output directory
    python augment_dataset.py --input dataset/images/train --labels dataset/labels/train \
    --output-images dataset/images/train_aug --output-labels dataset/labels/train_aug \
    --level heavy --multiplier 10
    # Preview augmentations without saving
    python augment_dataset.py --input dataset/images/train --labels dataset/labels/train --preview

OLD EXAMPLES / SETUP
--------------------
# scripts to run
# Opening Environment   - .\defect_env_gpu311\Scripts\activate
# Directing to scripts - cd scripts
# Augmenting dataset - python augment_dataset.py
=============================================================
"""

import os
import sys
import cv2
import glob
import shutil
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Error: albumentations not installed. Run: pip install albumentations>=1.3.0")


# ============================================================
# YOLO LABEL I/O
# ============================================================

def read_yolo_labels(label_path: str) -> List[List[float]]:
    """
    Read YOLO-format label file.
    Supports both bounding-box (5 values) and polygon/segmentation (variable values) formats.

    Returns:
        List of [class_id, x1, y1, x2, y2, ...] — all coords clamped to [0, 1]
    """
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                class_id = int(parts[0])
                # Read ALL coordinate values (works for both bbox and polygon)
                coords = [max(0.0, min(1.0, float(v))) for v in parts[1:]]
                labels.append([class_id] + coords)
    return labels


def write_yolo_labels(label_path: str, labels: List[List[float]]):
    """Write YOLO-format label file, preserving all coords (bbox or polygon)."""
    with open(label_path, 'w') as f:
        for label in labels:
            class_id = int(label[0])
            coords = ' '.join(f'{v:.6f}' for v in label[1:])
            f.write(f'{class_id} {coords}\n')


# ============================================================
# AUGMENTATION PIPELINES
# ============================================================

def get_augmentation_pipeline(level: str = "medium") -> 'A.Compose':
    """
    Build an augmentation pipeline suited for factory defect detection.
    Uses keypoint_params so that polygon vertices are transformed correctly
    (flips, rotations) — works for both bbox and segmentation label formats.

    Applies:
        - Rotation:          Up to ±15° (simulates part placement angle variation)
        - HorizontalFlip:    Mirror image (parts can appear mirrored)
        - VerticalFlip:      Upside-down flip (parts can be placed inverted)
        - GaussNoise:        Subtle sensor noise (simulates camera noise)
        - GaussianBlur:      Slight defocus blur (simulates camera focus variation)

    Note: CLAHE is intentionally excluded — images are already greyscaled with
    rust-aware CLAHE applied during capture (camera_capture.py).
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations is required for augmentation")

    transforms = [
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    ]

    pipeline = A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(
            format='xy',                 # absolute pixel coords
            label_fields=['kp_labels'],  # tracks which annotation each point belongs to
            remove_invisible=True,       # drop points that go outside the image
        )
    )
    return pipeline


# ============================================================
# AUGMENTATION ENGINE
# ============================================================

def augment_single_image(
    image_path: str,
    label_path: str,
    output_images_dir: str,
    output_labels_dir: str,
    pipeline: 'A.Compose',
    multiplier: int = 5,
    start_index: int = 0
) -> int:
    """
    Generate augmented copies of a single image + label pair.
    Preserves polygon annotations by transforming each vertex through
    the augmentation pipeline (flip, rotate, etc.).

    Returns:
        Number of successfully generated augmentations
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"  WARNING: Cannot read image: {image_path}")
        return 0

    h, w = image.shape[:2]
    labels = read_yolo_labels(label_path)

    # Flatten all polygon vertices into a single keypoints list.
    # kp_labels tracks which annotation index each keypoint belongs to
    # so we can reconstruct the polygons after augmentation.
    all_keypoints = []
    kp_labels = []
    ann_class_ids = []
    for ann_idx, lbl in enumerate(labels):
        ann_class_ids.append(int(lbl[0]))
        coords = lbl[1:]  # [x1, y1, x2, y2, ...] normalized
        for i in range(0, len(coords) - 1, 2):
            all_keypoints.append((coords[i] * w, coords[i + 1] * h))
            kp_labels.append(ann_idx)

    stem = Path(image_path).stem
    ext = Path(image_path).suffix
    generated = 0

    for i in range(multiplier):
        try:
            augmented = pipeline(
                image=image,
                keypoints=all_keypoints,
                kp_labels=kp_labels
            )

            aug_img = augmented['image']
            aug_h, aug_w = aug_img.shape[:2]
            aug_kps = augmented['keypoints']
            aug_kp_labels = augmented['kp_labels']

            # Group transformed keypoints back per annotation
            ann_points: dict = {}
            for pt, ann_idx in zip(aug_kps, aug_kp_labels):
                ann_points.setdefault(ann_idx, []).append(pt)

            # Rebuild label lines — skip annotations with <3 visible points
            aug_labels = []
            for ann_idx, pts in ann_points.items():
                if len(pts) < 3:
                    continue
                class_id = ann_class_ids[int(ann_idx)]  # cast: albumentations may return float
                coords = []
                for x, y in pts:
                    coords.append(max(0.0, min(1.0, x / aug_w)))
                    coords.append(max(0.0, min(1.0, y / aug_h)))
                aug_labels.append([class_id] + coords)

            # Build output filename
            aug_name = f"{stem}_aug{start_index + i:03d}"
            aug_image_path = os.path.join(output_images_dir, f"{aug_name}{ext}")
            aug_label_path = os.path.join(output_labels_dir, f"{aug_name}.txt")

            cv2.imwrite(aug_image_path, aug_img)
            write_yolo_labels(aug_label_path, aug_labels)
            generated += 1

        except Exception as e:
            print(f"  WARNING: Augmentation {i} failed for {stem}: {e}")

    return generated


def augment_dataset(
    images_dir: str,
    labels_dir: str,
    output_images_dir: Optional[str] = None,
    output_labels_dir: Optional[str] = None,
    multiplier: int = 2,
    level: str = "medium",
    copy_originals: bool = True,
    seed: int = 42
) -> dict:
    """
    Augment an entire dataset directory.
    
    Args:
        images_dir: Input images directory
        labels_dir: Input labels directory (YOLO format .txt files)
        output_images_dir: Output directory for augmented images (default: in-place)
        output_labels_dir: Output directory for augmented labels (default: in-place)
        multiplier: Number of augmented copies per image
        level: Augmentation intensity ('light', 'medium', 'heavy')
        copy_originals: If True, copy original images to output dir too
        seed: Random seed for reproducibility
        
    Returns:
        Summary dict with counts
    """
    random.seed(seed)
    np.random.seed(seed)

    # Default: augment in-place (add to same directory)
    if output_images_dir is None:
        output_images_dir = images_dir
    if output_labels_dir is None:
        output_labels_dir = labels_dir

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
    image_paths.sort()

    if not image_paths:
        print(f"ERROR: No images found in {images_dir}")
        return {'total_original': 0, 'total_augmented': 0, 'errors': 0}

    pipeline = get_augmentation_pipeline(level)

    print(f"\n{'='*60}")
    print(f"OFFLINE DATASET AUGMENTATION")
    print(f"{'='*60}")
    print(f"Source images:   {images_dir}")
    print(f"Source labels:   {labels_dir}")
    print(f"Output images:   {output_images_dir}")
    print(f"Output labels:   {output_labels_dir}")
    print(f"Images found:    {len(image_paths)}")
    print(f"Multiplier:      {multiplier}x")
    print(f"Level:           {level}")
    print(f"Expected output: ~{len(image_paths) * multiplier} augmented images")
    print(f"{'='*60}\n")

    # Optionally copy originals to output
    if copy_originals and output_images_dir != images_dir:
        print("Copying original images to output directory...")
        for img_path in image_paths:
            stem = Path(img_path).stem
            shutil.copy2(img_path, os.path.join(output_images_dir, Path(img_path).name))
            label_path = os.path.join(labels_dir, f"{stem}.txt")
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(output_labels_dir, f"{stem}.txt"))

    total_augmented = 0
    errors = 0

    for idx, img_path in enumerate(image_paths):
        stem = Path(img_path).stem
        label_path = os.path.join(labels_dir, f"{stem}.txt")

        if not os.path.exists(label_path):
            print(f"  [{idx+1}/{len(image_paths)}] SKIP {stem} — no label file")
            errors += 1
            continue

        n_generated = augment_single_image(
            image_path=img_path,
            label_path=label_path,
            output_images_dir=output_images_dir,
            output_labels_dir=output_labels_dir,
            pipeline=pipeline,
            multiplier=multiplier,
            start_index=idx * multiplier
        )

        total_augmented += n_generated
        print(f"  [{idx+1}/{len(image_paths)}] {stem} → {n_generated} augmented copies")

    print(f"\n{'='*60}")
    print(f"AUGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"Original images:  {len(image_paths)}")
    print(f"Augmented images: {total_augmented}")
    print(f"Total dataset:    {len(image_paths) + total_augmented} images")
    print(f"Skipped/errors:   {errors}")
    print(f"{'='*60}")

    return {
        'total_original': len(image_paths),
        'total_augmented': total_augmented,
        'errors': errors
    }


# ============================================================
# PREVIEW MODE — Visualize without saving
# ============================================================

def preview_augmentations(
    images_dir: str,
    labels_dir: str,
    level: str = "medium",
    num_samples: int = 3,
    augmentations_per_sample: int = 4
):
    """
    Preview augmented images in an OpenCV window (no files saved).
    
    Args:
        images_dir: Input images directory
        labels_dir: Input labels directory
        level: Augmentation intensity
        num_samples: Number of random images to preview
        augmentations_per_sample: Number of augmentations to show per image
    """
    # Find images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))

    if not image_paths:
        print("No images found for preview.")
        return

    # Pick random samples
    samples = random.sample(image_paths, min(num_samples, len(image_paths)))
    pipeline = get_augmentation_pipeline(level)

    # Class colors for drawing bboxes
    colors = [(0, 255, 0), (0, 0, 255), (255, 165, 0), (255, 0, 255), (0, 255, 255)]

    print(f"\nPREVIEW MODE — Showing {len(samples)} images × {augmentations_per_sample} augmentations each")
    print("Press any key to advance, 'q' to quit.\n")

    for img_path in samples:
        stem = Path(img_path).stem
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        image = cv2.imread(img_path)
        if image is None:
            continue

        labels = read_yolo_labels(label_path) if os.path.exists(label_path) else []
        h, w = image.shape[:2]

        # Derive bboxes from polygon points for visualization (min/max of x,y)
        def poly_to_bbox(coords):
            """Convert polygon coords [x1,y1,...] → yolo bbox [xc,yc,bw,bh]"""
            xs = coords[0::2]
            ys = coords[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            return [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]

        class_ids = [int(lbl[0]) for lbl in labels]
        bboxes = [poly_to_bbox(lbl[1:]) for lbl in labels]

        # Flatten polygon vertices for keypoint-based augmentation
        all_keypoints = []
        kp_labels = []
        for ann_idx, lbl in enumerate(labels):
            coords = lbl[1:]
            for i in range(0, len(coords) - 1, 2):
                all_keypoints.append((coords[i] * w, coords[i + 1] * h))
                kp_labels.append(ann_idx)

        # Draw original
        original_viz = draw_bboxes(image.copy(), bboxes, class_ids, colors)
        cv2.imshow(f"Original: {stem}", cv2.resize(original_viz, (640, 480)))

        for i in range(augmentations_per_sample):
            try:
                augmented = pipeline(image=image, keypoints=all_keypoints, kp_labels=kp_labels)
                aug_img = augmented['image']
                aug_h, aug_w = aug_img.shape[:2]
                # Group keypoints back and derive bboxes for display
                ann_pts: dict = {}
                for pt, ann_idx in zip(augmented['keypoints'], augmented['kp_labels']):
                    ann_pts.setdefault(ann_idx, []).append(pt)
                aug_bboxes, aug_cls = [], []
                for ann_idx, pts in ann_pts.items():
                    if len(pts) < 2:
                        continue
                    norm_coords = [v for pt in pts for v in (pt[0]/aug_w, pt[1]/aug_h)]
                    aug_bboxes.append(poly_to_bbox(norm_coords))
                    aug_cls.append(class_ids[ann_idx])
                aug_viz = draw_bboxes(aug_img.copy(), aug_bboxes, aug_cls, colors)
                cv2.imshow(f"Augmented #{i+1}: {stem}", cv2.resize(aug_viz, (640, 480)))
            except Exception as e:
                print(f"  Augmentation failed: {e}")

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Preview complete.")


def draw_bboxes(image: np.ndarray, bboxes: list, class_ids: list, colors: list) -> np.ndarray:
    """Draw YOLO bounding boxes on an image for visualization."""
    h, w = image.shape[:2]
    for bbox, cls_id in zip(bboxes, class_ids):
        xc, yc, bw, bh = bbox
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        color = colors[int(cls_id) % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"cls:{cls_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image


# ============================================================
# CLI
# ============================================================

def main():
    """CLI entry point for offline augmentation."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Offline dataset augmentation for YOLO defect detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Augment training set with 2x copies (medium intensity)
  python augment_dataset.py --input dataset/images/train --labels dataset/labels/train --multiplier 2

  # Heavy augmentation to separate output directory
  python augment_dataset.py --input dataset/images/train --labels dataset/labels/train \\
      --output-images dataset/images/train_aug --output-labels dataset/labels/train_aug \\
      --level heavy --multiplier 10

  # Preview augmentations without saving
  python augment_dataset.py --input dataset/images/train --labels dataset/labels/train --preview
        """
    )

    parser.add_argument('--input', default='dataset/images/train',
                        help='Input images directory (default: dataset/images/train)')
    parser.add_argument('--labels', default='dataset/labels/train',
                        help='Input labels directory (default: dataset/labels/train)')
    parser.add_argument('--output-images', default=None,
                        help='Output images directory (default: same as input — in-place)')
    parser.add_argument('--output-labels', default=None,
                        help='Output labels directory (default: same as labels — in-place)')
    parser.add_argument('--multiplier', type=int, default=3,
                        help='Number of augmented copies per image (default: 2)')
    parser.add_argument('--level', choices=['light', 'medium', 'heavy'], default='medium',
                        help='Augmentation intensity (default: medium)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--preview', action='store_true',
                        help='Preview augmentations in a window (no files saved)')
    parser.add_argument('--no-copy-originals', action='store_true',
                        help='Do not copy original images to output dir')

    args = parser.parse_args()

    if not ALBUMENTATIONS_AVAILABLE:
        print("\nERROR: albumentations is required.")
        print("Install with: pip install albumentations>=1.3.0")
        sys.exit(1)

    if args.preview:
        preview_augmentations(
            images_dir=args.input,
            labels_dir=args.labels,
            level=args.level
        )
    else:
        augment_dataset(
            images_dir=args.input,
            labels_dir=args.labels,
            output_images_dir=args.output_images,
            output_labels_dir=args.output_labels,
            multiplier=args.multiplier,
            level=args.level,
            copy_originals=not args.no_copy_originals,
            seed=args.seed
        )


if __name__ == "__main__":
    main()

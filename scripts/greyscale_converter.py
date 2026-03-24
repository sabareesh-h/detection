"""
Rust Area Darkener — YOLO Dataset Preprocessor
================================================
Darkens rust regions in part images so YOLO can clearly distinguish
rust from clean metal surfaces during training.

Usage:
    # Single image
    python darken_rust.py --input part.jpg

    # Whole dataset folder
    python darken_rust.py --input dataset/images/ --output dataset/images_processed/

    # Adjust darkening strength (0.0 = no change, 1.0 = pure black)
    python darken_rust.py --input dataset/images/ --strength 0.6

Requirements:
    pip install opencv-python numpy
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def build_rust_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Returns a float [0, 1] mask where 1.0 = strong rust, 0.0 = clean metal.
    Uses relative red-channel dominance so it works even under red-tinted lighting.
    """
    r = img_bgr[:, :, 2].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    b = img_bgr[:, :, 0].astype(np.float32)

    ratio = r / (g + b + 1.0)
    mean_r = ratio.mean()

    mask = np.clip((ratio - mean_r) / mean_r, 0, 1).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (7, 7), 2)   # smooth mask edges
    return mask


def darken_rust(img_bgr: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """
    Darkens rust regions in the image.

    Args:
        img_bgr:  Input image in BGR format.
        strength: How much to darken rust areas.
                  0.0 = no change | 0.55 = recommended | 1.0 = pitch black

    Returns:
        Greyscale image (BGR) with rust areas darkened.
    """
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Mild CLAHE to normalise brightness before darkening
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    grey = clahe.apply(grey)

    rust_mask = build_rust_mask(img_bgr)

    # Darken: rust pixels multiplied by (1 - strength), clean pixels unchanged
    dark_factor = 1.0 - rust_mask * strength
    result = np.clip(grey.astype(np.float32) * dark_factor, 0, 255).astype(np.uint8)

    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def process(src: Path, dst_dir: Path, strength: float):
    img = cv2.imread(str(src))
    if img is None:
        print(f"  [SKIP] Cannot read: {src.name}")
        return
    out = dst_dir / f"{src.stem}_processed.png"
    dst_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), darken_rust(img, strength))
    print(f"  ✓  {src.name}  →  {out.name}")


def main():
    parser = argparse.ArgumentParser(description="Darken rust areas for YOLO training")
    parser.add_argument("--input",    "-i", required=True,  help="Image file or folder")
    parser.add_argument("--output",   "-o", default=None,   help="Output folder")
    parser.add_argument("--strength", "-s", type=float, default=0.55,
                        help="Darkening strength 0.0–1.0 (default: 0.55)")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"Error: not found — {src}")
        sys.exit(1)

    dst = Path(args.output) if args.output else src.parent / f"{src.stem}_darkened"

    files = (
        [src] if src.is_file()
        else [f for f in sorted(src.rglob("*")) if f.suffix.lower() in SUPPORTED]
    )

    print(f"Processing {len(files)} image(s) | strength={args.strength}")
    for f in files:
        rel = f.relative_to(src) if src.is_dir() else Path(f.name)
        process(f, dst / rel.parent, args.strength)

    print(f"\nDone → {dst}/")


if __name__ == "__main__":
    main()
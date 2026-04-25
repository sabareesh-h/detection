"""
=============================================================
  add_data.py  –  New-Data Intake Wizard
=============================================================
HOW TO USE
----------
1.  Put your new images into:   scripts/new_data/images/
2.  Put your new labels into:   scripts/new_data/labels/
    (YOLO .txt format — one file per image, same stem)

3.  Run:
        python add_data.py

    The script will:
        a) Validate every image/label pair
        b) Copy them into  dataset/raw/images  and  dataset/raw/labels
        c) Re-split the FULL raw pool  →  train / val / test
        d) (Optional) Run augmentation on the new training images

4.  Then train as usual:
        python train_model.py --preset good_vs_rust_optimized --name my_run --no-wandb

FLAGS
-----
    --inbox   PATH   Override the inbox folder (default: new_data/)
    --split-only     Skip inbox; just re-split the existing raw pool
    --no-augment     Skip the augmentation step
    --multiplier N   Augmentation copies per image (default: 3)
    --dry-run        Show what would happen without making changes
=============================================================
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path

# ── project layout ──────────────────────────────────────────────────────────
SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent

INBOX_DIR    = SCRIPTS_DIR / "new_data"          # drop zone
RAW_DIR      = PROJECT_ROOT / "dataset" / "raw"  # accumulated originals
DATASET_DIR  = PROJECT_ROOT / "dataset"          # images/ labels/ live here

RAW_IMAGES   = RAW_DIR / "images"
RAW_LABELS   = RAW_DIR / "labels"

# ── split ratios (must sum to 1.0) ──────────────────────────────────────────
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
RANDOM_SEED  = 42

IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _hr(char="=", width=60):
    print(char * width)


def _info(msg):  print(f"  {msg}")
def _ok(msg):    print(f"  [OK]  {msg}")
def _warn(msg):  print(f"  [!!]  {msg}")
def _err(msg):   print(f"  [XX]  {msg}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 – Validate & copy inbox → raw
# ════════════════════════════════════════════════════════════════════════════

def ingest_inbox(inbox: Path, dry_run: bool = False) -> dict:
    """
    Copy every valid image+label pair from inbox into RAW_IMAGES / RAW_LABELS.
    Returns counts of accepted, skipped, and duplicate files.
    """
    inbox_images = inbox / "images"
    inbox_labels = inbox / "labels"

    _hr()
    print("STEP 1 -- Ingesting new data from inbox")
    _hr()
    _info(f"Inbox images : {inbox_images}")
    _info(f"Inbox labels : {inbox_labels}")

    # ── quick sanity checks ────────────────────────────────────────────────
    if not inbox_images.exists():
        _err(f"Inbox images folder not found: {inbox_images}")
        _err("Create it and put your new .png / .jpg files there.")
        sys.exit(1)
    if not inbox_labels.exists():
        _err(f"Inbox labels folder not found: {inbox_labels}")
        _err("Create it and put your YOLO .txt label files there.")
        sys.exit(1)

    all_images = [f for f in inbox_images.iterdir()
                  if f.suffix.lower() in IMAGE_EXTS]

    if not all_images:
        _warn("No images found in the inbox. Nothing to add.")
        return {"accepted": 0, "no_label": 0, "duplicate": 0, "total": 0}

    _info(f"Found {len(all_images)} image(s) in inbox.\n")

    if not dry_run:
        RAW_IMAGES.mkdir(parents=True, exist_ok=True)
        RAW_LABELS.mkdir(parents=True, exist_ok=True)

    accepted = skipped_no_label = skipped_duplicate = 0

    for img_path in sorted(all_images):
        stem = img_path.stem
        lbl_path = inbox_labels / f"{stem}.txt"

        # ── check label exists ────────────────────────────────────────────
        if not lbl_path.exists():
            _warn(f"No label found for  {img_path.name}  → SKIPPED")
            skipped_no_label += 1
            continue

        # ── check not a duplicate in raw ──────────────────────────────────
        dest_img = RAW_IMAGES / img_path.name
        dest_lbl = RAW_LABELS / lbl_path.name
        if dest_img.exists():
            _warn(f"Already in raw:  {img_path.name}  → SKIPPED (duplicate)")
            skipped_duplicate += 1
            continue

        # ── copy ──────────────────────────────────────────────────────────
        if not dry_run:
            shutil.copy2(img_path, dest_img)
            shutil.copy2(lbl_path, dest_lbl)
        _ok(f"{img_path.name}  +  {lbl_path.name}")
        accepted += 1

    print()
    _hr("-")
    print(f"  Accepted  : {accepted}")
    print(f"  No label  : {skipped_no_label}")
    print(f"  Duplicate : {skipped_duplicate}")
    _hr("-")
    return {"accepted": accepted,
            "no_label": skipped_no_label,
            "duplicate": skipped_duplicate,
            "total": len(all_images)}


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 – Re-split the full raw pool
# ════════════════════════════════════════════════════════════════════════════

def split_dataset(dry_run: bool = False):
    """
    Clear train / val / test folders and rebuild them from the full raw pool.
    This guarantees a clean, consistent split every time data is added.
    """
    _hr()
    print("STEP 2 -- Re-splitting raw pool -> train / val / test")
    _hr()

    image_files = sorted([
        f for f in RAW_IMAGES.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    ])

    if not image_files:
        _err(f"No images found in raw pool: {RAW_IMAGES}")
        sys.exit(1)

    _info(f"Total images in raw pool : {len(image_files)}")
    _info(f"Split  ->  train {TRAIN_RATIO:.0%}  val {VAL_RATIO:.0%}  test {TEST_RATIO:.0%}")

    random.seed(RANDOM_SEED)
    random.shuffle(image_files)

    n = len(image_files)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": image_files[:n_train],
        "val":   image_files[n_train : n_train + n_val],
        "test":  image_files[n_train + n_val:],
    }

    for split_name, files in splits.items():
        img_dir = DATASET_DIR / "images" / split_name
        lbl_dir = DATASET_DIR / "labels" / split_name

        if not dry_run:
            # Wipe old split so stale files don't linger
            if img_dir.exists(): shutil.rmtree(img_dir)
            if lbl_dir.exists(): shutil.rmtree(lbl_dir)
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_file in files:
                shutil.copy2(img_file, img_dir / img_file.name)
                lbl_file = RAW_LABELS / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    shutil.copy2(lbl_file, lbl_dir / lbl_file.name)
                else:
                    _warn(f"No label in raw for {img_file.name}")

        _ok(f"{split_name:6s}: {len(files)} images")

    print()


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 – Augment only the NEW training images
# ════════════════════════════════════════════════════════════════════════════

def augment_new_images(new_stems: list, multiplier: int = 3, dry_run: bool = False):
    """
    Run augmentation on the new images that ended up in the training split.
    Only augments the newly added files (not the whole train set again).
    """
    _hr()
    print("STEP 3 -- Augmenting new training images")
    _hr()

    try:
        # Import the augmentation function from the sibling script
        sys.path.insert(0, str(SCRIPTS_DIR))
        from augment_dataset import augment_single_image, get_augmentation_pipeline
    except ImportError as e:
        _warn(f"Could not import augment_dataset.py: {e}")
        _warn("Skipping augmentation. Run  python augment_dataset.py  manually.")
        return

    train_img_dir = DATASET_DIR / "images" / "train"
    train_lbl_dir = DATASET_DIR / "labels" / "train"

    # Which of the new images actually ended up in train?
    to_augment = [
        train_img_dir / f"{stem}{ext}"
        for stem in new_stems
        for ext in IMAGE_EXTS
        if (train_img_dir / f"{stem}{ext}").exists()
    ]

    if not to_augment:
        _info("None of the new images landed in the training split.")
        _info("(They may all be in val/test — try adding more data.)")
        return

    _info(f"Augmenting {len(to_augment)} new training image(s) × {multiplier} copies each.")
    _info(f"Expected new augmented files: ~{len(to_augment) * multiplier}\n")

    if dry_run:
        for p in to_augment:
            _info(f"[dry-run] would augment: {p.name}")
        return

    pipeline = get_augmentation_pipeline("medium")
    total_generated = 0

    for idx, img_path in enumerate(to_augment):
        stem = img_path.stem
        lbl_path = train_lbl_dir / f"{stem}.txt"

        if not lbl_path.exists():
            _warn(f"No label for {img_path.name} in train/labels — skipping")
            continue

        n = augment_single_image(
            image_path=str(img_path),
            label_path=str(lbl_path),
            output_images_dir=str(train_img_dir),
            output_labels_dir=str(train_lbl_dir),
            pipeline=pipeline,
            multiplier=multiplier,
            start_index=idx * multiplier,
        )
        _ok(f"{img_path.name}  ->  {n} augmented copies")
        total_generated += n

    print()
    _hr("-")
    _info(f"Total augmented images added: {total_generated}")
    _hr("-")


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 – Clear the inbox (optional prompt)
# ════════════════════════════════════════════════════════════════════════════

def offer_clear_inbox(inbox: Path, dry_run: bool):
    print()
    _hr()
    print("STEP 4 -- Clear inbox?")
    _hr()
    _info("Your new files have been copied into the dataset.")
    answer = input("  Delete contents of inbox now? [y/N]: ").strip().lower()
    if answer == "y":
        if not dry_run:
            for sub in ["images", "labels"]:
                d = inbox / sub
                if d.exists():
                    shutil.rmtree(d)
                    d.mkdir()
        _ok("Inbox cleared.")
    else:
        _info("Inbox left as-is.")


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def print_summary(stats: dict, dry_run: bool):
    print()
    _hr()
    print("DONE" + ("  [DRY RUN -- no files changed]" if dry_run else ""))
    _hr()
    raw_total = len(list(RAW_IMAGES.glob("*"))) if RAW_IMAGES.exists() else "?"
    _info(f"Raw pool now contains   : {raw_total} images")

    for split in ("train", "val", "test"):
        d = DATASET_DIR / "images" / split
        count = len(list(d.glob("*"))) if d.exists() else 0
        _info(f"  {split:6s} images     : {count}")

    print()
    _info("Next step -> train the model:")
    _info("  python train_model.py --preset good_vs_rust_optimized --name my_run --no-wandb")
    _hr()


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Add new training data and re-split the dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--inbox",       default=str(INBOX_DIR),
                        help=f"Inbox folder (default: {INBOX_DIR})")
    parser.add_argument("--split-only",  action="store_true",
                        help="Skip inbox; just re-split the existing raw pool")
    parser.add_argument("--no-augment",  action="store_true",
                        help="Skip the augmentation step")
    parser.add_argument("--multiplier",  type=int, default=3,
                        help="Augmented copies per new image (default: 3)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Show what would happen without making any changes")
    args = parser.parse_args()

    inbox = Path(args.inbox)

    print()
    _hr()
    print(" NEW DATA INTAKE WIZARD")
    _hr()
    if args.dry_run:
        print("  *** DRY RUN -- no files will be changed ***")
        print()

    # ── Step 1: ingest ──────────────────────────────────────────────────────
    new_stems = []
    stats = {}

    if not args.split_only:
        # Create inbox subfolders with a README if they don't exist yet
        for sub in ("images", "labels"):
            (inbox / sub).mkdir(parents=True, exist_ok=True)

        stats = ingest_inbox(inbox, dry_run=args.dry_run)
        if stats["accepted"] == 0 and not args.dry_run:
            print()
            _info("Nothing new to add. Use  --split-only  to just re-split.")
            print()
            # Still offer to continue to a re-split if raw has data
            if not any(RAW_IMAGES.glob("*")):
                sys.exit(0)

        # Remember which stems were newly accepted (for targeted augmentation)
        inbox_images = inbox / "images"
        new_stems = [
            f.stem for f in inbox_images.iterdir()
            if f.suffix.lower() in IMAGE_EXTS and
               (RAW_IMAGES / f.name).exists()   # means it was accepted
        ] if inbox_images.exists() else []

    # ── Step 2: split ───────────────────────────────────────────────────────
    split_dataset(dry_run=args.dry_run)

    # ── Step 3: augment new training images ─────────────────────────────────
    if not args.no_augment and new_stems:
        augment_new_images(new_stems, multiplier=args.multiplier, dry_run=args.dry_run)
    elif not args.no_augment and args.split_only:
        _info("--split-only: skipping augmentation.")

    # ── Step 4: offer to clear inbox ────────────────────────────────────────
    if not args.split_only and stats.get("accepted", 0) > 0 and not args.dry_run:
        offer_clear_inbox(inbox, dry_run=args.dry_run)

    # ── Summary ─────────────────────────────────────────────────────────────
    print_summary(stats, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

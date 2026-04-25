# New Data Inbox

This is the **drop zone** for new training images and labels.

## How to add new data

1. Copy your **images** (`.png` / `.jpg`) into:
   ```
   new_data/images/
   ```

2. Copy the matching **YOLO label files** (`.txt`) into:
   ```
   new_data/labels/
   ```
   > Each label file must have the **same filename** as its image, e.g.  
   > `scratch_001.png` → `scratch_001.txt`

3. Run the intake wizard from the `scripts/` folder:
   ```bash
   cd scripts
   python add_data.py
   ```

That's it. The script will:
- ✔ Validate every image/label pair
- ✔ Copy accepted pairs into `dataset/raw/`
- ✔ Re-split the full raw pool → `train / val / test`
- ✔ Augment only the newly added training images
- ✔ Offer to clear the inbox when done

Then train as usual:
```bash
python train_model.py --preset good_vs_rust_optimized --name my_run --no-wandb
```

## Flags

| Flag | Effect |
|------|--------|
| `--split-only` | Skip inbox; just re-split the existing raw pool |
| `--no-augment` | Skip the augmentation step |
| `--multiplier N` | Augmented copies per image (default: 3) |
| `--dry-run` | Show what would happen without making changes |

# Model Evaluation Report

**Generated**: 2026-03-17T08:24:22.364444

## Model Info

| Property | Value |
|----------|-------|
| Model | `runs/detect/models/defect_yolo26m_20260315_135621/weights/best.pt` |
| Dataset | `../config/dataset.yaml` |
| Split | `test` |
| Image Size | `640` |
| Confidence | `0.5` |
| IoU | `0.45` |
| Device | `NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU` |

## Overall Metrics

| Metric | Value |
|--------|-------|
| mAP@50 | **0.982** |
| mAP@50-95 | **0.8475** |
| Precision | 0.9679 |
| Recall | 0.9693 |
| F1 Score | 0.9686 |

## Per-Class Metrics

| Class | AP@50 | AP@50-95 | Precision | Recall | F1 |
|-------|-------|----------|-----------|--------|-----|
| Good(Top) | 0.9919 | 0.9213 | 0.9652 | 0.9907 | 0.9778 |
| Rust(Top) | 0.9636 | 0.8909 | 0.9661 | 0.9344 | 0.95 |
| Rust(Mid) | 0.9819 | 0.8643 | 0.9552 | 0.9697 | 0.9624 |
| Rust(Bottom) | 0.9548 | 0.7966 | 0.9394 | 0.9254 | 0.9323 |
| Rust(Thread) | 0.99 | 0.7926 | 0.9818 | 0.9818 | 0.9818 |
| Good(Mid) | 0.9902 | 0.9264 | 0.9906 | 0.9805 | 0.9855 |
| Good(Thread) | 0.9948 | 0.7968 | 0.9725 | 0.9907 | 0.9815 |
| Good(Bottom) | 0.9885 | 0.7907 | 0.9722 | 0.9813 | 0.9767 |

## Inference Speed

| Metric | Value |
|--------|-------|
| FPS | **53.0** |
| Total (mean) | 18.86 ms |
| Total (std) | ±1.03 ms |
| Preprocess | 1.94 ms |
| Inference | 16.46 ms |
| Postprocess | 0.3 ms |
| Device | NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU |

## Generated Files

- `evaluation_report.json` — Full metrics data
- `confidence_histogram.png` — Confidence score distribution
- `val_results/confusion_matrix.png` — Confusion matrix
- `val_results/confusion_matrix_normalized.png` — Normalized confusion matrix
- `val_results/` — Full Ultralytics validation output
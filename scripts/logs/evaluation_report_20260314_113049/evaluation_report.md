# Model Evaluation Report

**Generated**: 2026-03-14T11:30:49.215631

## Model Info

| Property | Value |
|----------|-------|
| Model | `runs/detect/models/defect_yolo26m_20260314_095127/weights/best.pt` |
| Dataset | `../config/dataset.yaml` |
| Split | `test` |
| Image Size | `640` |
| Confidence | `0.5` |
| IoU | `0.45` |
| Device | `NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU` |

## Overall Metrics

| Metric | Value |
|--------|-------|
| mAP@50 | **0.9896** |
| mAP@50-95 | **0.857** |
| Precision | 0.9765 |
| Recall | 0.9877 |
| F1 Score | 0.9821 |

## Per-Class Metrics

| Class | AP@50 | AP@50-95 | Precision | Recall | F1 |
|-------|-------|----------|-----------|--------|-----|
| Good(Top) | 0.995 | 0.9108 | 1.0 | 1.0 | 1.0 |
| Rust(Top) | 0.9911 | 0.9378 | 1.0 | 0.9836 | 0.9917 |
| Rust(Mid) | 0.9943 | 0.8962 | 0.9565 | 1.0 | 0.9778 |
| Rust(Bottom) | 0.9692 | 0.8409 | 0.9697 | 0.9552 | 0.9624 |
| Rust(Thread) | 0.9873 | 0.7886 | 0.931 | 0.9818 | 0.9557 |
| Good(Mid) | 0.9903 | 0.893 | 1.0 | 0.9813 | 0.9906 |
| Good(Thread) | 0.995 | 0.8013 | 0.9817 | 1.0 | 0.9908 |
| Good(Bottom) | 0.995 | 0.7876 | 0.9727 | 1.0 | 0.9862 |

## Inference Speed

| Metric | Value |
|--------|-------|
| FPS | **52.7** |
| Total (mean) | 18.99 ms |
| Total (std) | ±0.91 ms |
| Preprocess | 1.85 ms |
| Inference | 16.71 ms |
| Postprocess | 0.28 ms |
| Device | NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU |

## Generated Files

- `evaluation_report.json` — Full metrics data
- `confidence_histogram.png` — Confidence score distribution
- `val_results/confusion_matrix.png` — Confusion matrix
- `val_results/confusion_matrix_normalized.png` — Normalized confusion matrix
- `val_results/` — Full Ultralytics validation output
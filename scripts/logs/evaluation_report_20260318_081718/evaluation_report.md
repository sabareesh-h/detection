# Model Evaluation Report

**Generated**: 2026-03-18T08:17:18.663375

## Model Info

| Property | Value |
|----------|-------|
| Model | `runs/detect/models/defect_yolo26m_20260317_144141/weights/best.pt` |
| Dataset | `../config/dataset.yaml` |
| Split | `test` |
| Image Size | `1024` |
| Confidence | `0.5` |
| IoU | `0.45` |
| Device | `NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU` |

## Overall Metrics

| Metric | Value |
|--------|-------|
| mAP@50 | **0.9834** |
| mAP@50-95 | **0.8607** |
| Precision | 0.9786 |
| Recall | 0.9843 |
| F1 Score | 0.9814 |

## Per-Class Metrics

| Class | AP@50 | AP@50-95 | Precision | Recall | F1 |
|-------|-------|----------|-----------|--------|-----|
| Good(Top) | 0.9867 | 0.9644 | 1.0 | 0.9744 | 0.987 |
| Rust(Top) | 0.9735 | 0.93 | 1.0 | 0.9474 | 0.973 |
| Rust(Mid) | 0.9508 | 0.8537 | 0.9091 | 1.0 | 0.9524 |
| Rust(Bottom) | 0.976 | 0.8145 | 1.0 | 0.9524 | 0.9756 |
| Rust(Thread) | 0.995 | 0.7973 | 0.9444 | 1.0 | 0.9714 |
| Good(Mid) | 0.995 | 0.9459 | 0.975 | 1.0 | 0.9873 |
| Good(Thread) | 0.995 | 0.7803 | 1.0 | 1.0 | 1.0 |
| Good(Bottom) | 0.995 | 0.7996 | 1.0 | 1.0 | 1.0 |

## Inference Speed

| Metric | Value |
|--------|-------|
| FPS | **18.3** |
| Total (mean) | 54.79 ms |
| Total (std) | ±2.71 ms |
| Preprocess | 12.84 ms |
| Inference | 40.59 ms |
| Postprocess | 0.87 ms |
| Device | NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU |

## Generated Files

- `evaluation_report.json` — Full metrics data
- `confidence_histogram.png` — Confidence score distribution
- `val_results/confusion_matrix.png` — Confusion matrix
- `val_results/confusion_matrix_normalized.png` — Normalized confusion matrix
- `val_results/` — Full Ultralytics validation output
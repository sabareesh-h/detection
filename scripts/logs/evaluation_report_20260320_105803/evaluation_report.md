# Model Evaluation Report

**Generated**: 2026-03-20T10:58:03.790799

## Model Info

| Property | Value |
|----------|-------|
| Model | `runs/detect/models/defect_yolo26m_20260319_142148/weights/best.pt` |
| Dataset | `../config/dataset.yaml` |
| Split | `test` |
| Image Size | `1024` |
| Confidence | `0.5` |
| IoU | `0.45` |
| Device | `NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU` |

## Overall Metrics

| Metric | Value |
|--------|-------|
| mAP@50 | **0.9869** |
| mAP@50-95 | **0.8593** |
| Precision | 0.9843 |
| Recall | 0.9817 |
| F1 Score | 0.983 |

## Per-Class Metrics

| Class | AP@50 | AP@50-95 | Precision | Recall | F1 |
|-------|-------|----------|-----------|--------|-----|
| Good(Top) | 0.995 | 0.9574 | 1.0 | 1.0 | 1.0 |
| Rust(Top) | 0.995 | 0.9544 | 1.0 | 1.0 | 1.0 |
| Rust(Mid) | 0.9902 | 0.8812 | 0.9524 | 1.0 | 0.9756 |
| Rust(Bottom) | 0.9523 | 0.7905 | 1.0 | 0.9048 | 0.95 |
| Rust(Thread) | 0.995 | 0.7954 | 0.9444 | 1.0 | 0.9714 |
| Good(Mid) | 0.995 | 0.9398 | 1.0 | 1.0 | 1.0 |
| Good(Thread) | 0.9867 | 0.7716 | 1.0 | 0.9744 | 0.987 |
| Good(Bottom) | 0.9864 | 0.7841 | 0.9778 | 0.9744 | 0.9761 |

## Inference Speed

| Metric | Value |
|--------|-------|
| FPS | **20.2** |
| Total (mean) | 49.6 ms |
| Total (std) | ±2.97 ms |
| Preprocess | 5.69 ms |
| Inference | 43.35 ms |
| Postprocess | 0.35 ms |
| Device | NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU |

## Generated Files

- `evaluation_report.json` — Full metrics data
- `confidence_histogram.png` — Confidence score distribution
- `val_results/confusion_matrix.png` — Confusion matrix
- `val_results/confusion_matrix_normalized.png` — Normalized confusion matrix
- `val_results/` — Full Ultralytics validation output
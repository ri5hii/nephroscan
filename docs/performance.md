# Model Performance & Metrics

## Current Performance (YOLO11 Nano)
- mAP50: Optimized for medical accuracy
- Precision: High precision for clinical use
- Recall: Sensitive kidney stone detection
- Training Speed: ~45 minutes on RTX 3050 4GB
- Inference Speed: Real-time detection

## Memory Usage Benchmarks
| GPU Model      | Max Batch Size | Optimal Image Size | Training Time (50 epochs) |
|---------------|---------------|-------------------|---------------------------|
| RTX 3050 4GB  | 1             | 320px             | ~45 minutes               |
| RTX 4060 8GB  | 4-6           | 480px             | ~25 minutes               |
| RTX 4070 12GB | 8-12          | 640px             | ~15 minutes               |

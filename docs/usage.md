# Usage Guide

## Training Workflow
1. Prepare Dataset: Organize images and labels in YOLO format in `data/` directory
2. Optimize GPU: Run `./scripts/sh/optimize_gpu.sh` for memory optimization
3. Launch GUI: `python src/main.py`
4. Configure Training: Set epochs, batch size based on your GPU
5. Start Training: Monitor progress and memory usage in real-time
6. Checkpoint Management: Automatic cleanup maintains optimal storage

## Inference Workflow
1. Load Model: Select trained checkpoint through GUI
2. Input Selection: Choose single images or batch processing
3. Preprocessing: Automatic adaptive preprocessing applied
4. Detection: Real-time kidney stone detection with confidence scores
5. Results: Bounding boxes, confidence metrics, and medical analysis
6. Report Generation: Professional PDF reports for documentation

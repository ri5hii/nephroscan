# YOLOv8 Kidney Stone Detection Model - Version 3

## Overview
This is the most comprehensive version of the YOLOv8 kidney stone detection model, trained on an enhanced dataset with 1,300+ high-quality medical images.

## Model Information
- **Name**: YOLOv8 Kidney Stone Detection v3
- **Version**: 3.0
- **Framework**: YOLOv8 (Ultralytics)
- **Dataset Size**: 1,300 images
- **Task**: Medical Object Detection
- **Classes**: KidneyStone (Tas_Var)

## Dataset Details
### Enhanced Medical Dataset
- **Total Images**: 1,300
- **Training Set**: 1,054 images (81%)
- **Validation Set**: 123 images (9.5%)
- **Test Set**: 123 images (9.5%)

### Dataset Source
- **Provider**: Roboflow - Tez ROI Aug Dataset
- **License**: CC BY 4.0
- **Quality**: High-resolution medical images with professional annotations

## Directory Structure
```
yolov8_kidney_stone_v3/
├── configs/           # Configuration files
│   ├── data_v3.yaml              # Dataset configuration
│   ├── model_config_v3.yaml      # Model configuration
│   └── training_summary_*.yaml   # Training results
├── scripts/           # Training and inference scripts
│   ├── train_v3.py               # Enhanced training script
│   └── inference_v3.py           # Inference script
├── weights/           # Model weights
│   ├── best_v3_*.pt              # Best performing weights
│   └── last_v3_*.pt              # Latest weights
├── results/           # Training results and logs
│   └── run directories with metrics and plots
└── README.md         # This file
```

## Key Improvements in Version 3

### Dataset Enhancements
- **10x Larger Dataset**: 1,300 vs 114 images in previous versions
- **Proper Data Splits**: Professional train/validation/test division
- **Better Annotations**: High-quality medical annotations
- **Diverse Cases**: Wide variety of kidney stone presentations

### Model Improvements
- **Larger Model**: YOLOv8m (medium) for better performance
- **Higher Resolution**: 640x640 input size vs 224x224
- **Enhanced Training**: 100 epochs with optimized hyperparameters
- **Better Augmentation**: Medical-specific data augmentation

### Technical Optimizations
- **GPU Memory Management**: Adaptive batch sizing
- **Faster Training**: Enabled caching and more workers
- **Better Monitoring**: Enhanced logging and metrics tracking

## Usage

### Training
```bash
cd /home/rishi/Desktop/nephroscan/models/yolov8_kidney_stone_v3/scripts

# Quick training (50 epochs)
python train_v3.py --epochs 50

# Full training with optimal settings
python train_v3.py --model-size m --epochs 100 --batch-size 16 --img-size 640

# Training on systems with limited GPU memory
python train_v3.py --model-size s --batch-size 8 --img-size 416
```

### Training Parameters
- `--model-size`: YOLOv8 variant (n/s/m/l/x) - default: 'm' (medium)
- `--epochs`: Number of training epochs - default: 100
- `--batch-size`: Batch size - default: 16 (auto-adjusted for GPU memory)
- `--img-size`: Input image size - default: 640
- `--patience`: Early stopping patience - default: 15

### Hardware Requirements
- **Minimum GPU Memory**: 4GB (with reduced settings)
- **Recommended GPU Memory**: 8GB+
- **CUDA**: Required for GPU training
- **RAM**: Minimum 16GB recommended for large dataset

## Expected Performance
Based on the larger, higher-quality dataset, v3 is expected to achieve:
- **Higher mAP**: Improved mean Average Precision
- **Better Generalization**: More robust performance on diverse cases
- **Lower False Positives**: More precise detections
- **Higher Recall**: Better detection of smaller stones

## Model Comparison
| Version | Dataset Size | Model Size | Image Size | Expected Performance |
|---------|-------------|------------|------------|---------------------|
| v1      | 114 images  | YOLOv8s    | 224px     | Baseline            |
| v2      | 114 images  | YOLOv8s    | 224px     | Organized structure |
| v3      | 1,300 images| YOLOv8m    | 640px     | **Best performance**|

## Training Configuration

### Optimized Hyperparameters
- **Learning Rate**: 0.01 (higher for larger dataset)
- **Batch Size**: 16 (adaptive based on GPU memory)
- **Image Size**: 640x640 (standard YOLO resolution)
- **Augmentation**: Medical-specific (no vertical flips, minimal color changes)

### Medical-Specific Settings
- **No Mixup/Copy-Paste**: Inappropriate for medical imaging
- **Conservative Augmentation**: Preserves medical image characteristics
- **Higher Patience**: Allows convergence on complex medical patterns

## Evaluation Metrics
- **Primary**: mAP@0.5, mAP@0.5:0.95
- **Medical**: Sensitivity, Specificity, Precision, Recall
- **Clinical**: False Positive Rate, False Negative Rate

## Version History
- **v1.0**: Initial 114-image model
- **v2.0**: Organized structure, same dataset
- **v3.0**: **Current** - Enhanced 1,300-image dataset with optimized training

## Expected Training Time
- **RTX 3050 4GB**: ~8-12 hours for 100 epochs
- **RTX 4060 8GB**: ~4-6 hours for 100 epochs
- **RTX 4080 16GB**: ~2-3 hours for 100 epochs

## Notes
- Training automatically adjusts batch size based on available GPU memory
- Model saves checkpoints every 10 epochs for recovery
- Best weights are automatically saved when validation improves
- Comprehensive logging tracks all training metrics and hyperparameters
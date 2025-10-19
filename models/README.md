# Models Organization

This directory contains all model-related files for NephroScan.

## Structure

```
models/
 pretrained/                    # Pretrained YOLO models
    yolov8s.pt                # YOLOv8 small pretrained model
    yolo11n.pt                # YOLO11 nano pretrained model

 yolov8_kidney_stone_v3/        # Our trained kidney stone detection model
     configs/                   # Configuration files
        model_config_v3.yaml   # Main training configuration
        data_v3.yaml          # Dataset configuration
     weights/                   # Final model weights
        best_v3_*.pt          # Best performing model
        last_v3_*.pt          # Latest model checkpoint
        legacy_*.pt           # Backup/legacy weights
     scripts/                   # Model-specific scripts
     README.md                  # Model documentation
```

## File Types

- **pretrained/**: Downloaded pretrained models from Ultralytics
- **weights/**: Our trained model weights and checkpoints
- **configs/**: YAML configuration files for training and data
- **scripts/**: Model-specific utility scripts

## Training Results

Training results (logs, plots, intermediate weights) are stored in:
- `output/training_cycles/training_YYYYMMDD_HHMMSS/`

This keeps training artifacts separate from final model weights.
# NephroScan Configuration Guide

This document provides comprehensive configuration guidance for optimal performance across different hardware setups.

## GPU Configuration Profiles

### RTX 3050 4GB (Ultra Memory-Efficient)
```yaml
# Training parameters - Optimized for 4GB GPU
epochs: 25
imgsz: 320
batch: 1
patience: 15

# Hardware configuration
device: 'auto'
workers: 1
amp: false

# Memory optimization settings
max_det: 50
cache: false
half: false
pin_memory: false

# Learning rate configuration
lr0: 0.001
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
```

### RTX 4060 8GB (Balanced Performance)
```yaml
# Training parameters - Balanced for 8GB GPU
epochs: 50
imgsz: 480
batch: 4
patience: 20

# Hardware configuration
device: 'auto'
workers: 2
amp: false

# Memory optimization settings
max_det: 100
cache: false
half: false
pin_memory: false

# Learning rate configuration
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
```

### RTX 4070+ 12GB+ (High Performance)
```yaml
# Training parameters - Optimized for high-end GPUs
epochs: 100
imgsz: 640
batch: 8
patience: 30

# Hardware configuration
device: 'auto'
workers: 4
amp: true

# Memory optimization settings
max_det: 300
cache: true
half: false
pin_memory: true

# Learning rate configuration
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
```

## Environment Variables for Memory Optimization

### Essential Variables
```bash
# Memory allocator optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Debugging and monitoring
export CUDA_LAUNCH_BLOCKING=1

# Specify GPU device (single GPU systems)
export CUDA_VISIBLE_DEVICES=0

# Memory management
export PYTHONHASHSEED=42
```

### Memory Monitoring Commands
```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Clear GPU memory
python clear_gpu.py
```

## Adaptive Preprocessing Configuration

### CT Scan Detection Thresholds
```python
# Configuration in image_preprocessor.py
CT_SCAN_MIN_RESOLUTION = 640  # Minimum resolution for CT scan detection
CT_SCAN_CONTRAST_THRESHOLD = 0.3  # Contrast ratio threshold
CT_SCAN_BRIGHTNESS_RANGE = (50, 200)  # Brightness range for CT scans
```

### Quality Assessment Parameters
```python
# Low quality detection thresholds
LOW_QUALITY_CONTRAST_THRESHOLD = 0.15
MEDIUM_QUALITY_CONTRAST_THRESHOLD = 0.25
HIGH_QUALITY_CONTRAST_THRESHOLD = 0.4

# Noise detection parameters
NOISE_THRESHOLD = 0.1
BLUR_THRESHOLD = 100
```

### Preprocessing Methods
```python
# Standard medical preprocessing
TARGET_SIZE = (832, 832)  # For high-end GPUs
TARGET_SIZE_LOW_MEMORY = (320, 320)  # For RTX 3050 4GB

# Enhancement parameters
CLAHE_CLIP_LIMIT = 2.0
BILATERAL_FILTER_D = 9
GAUSSIAN_KERNEL_SIZE = (3, 3)
```

## Training Optimization Settings

### Loss Function Weights
```yaml
# Optimized for medical accuracy
box: 7.5       # Box regression loss weight
cls: 0.5       # Classification loss weight  
dfl: 1.5       # Distribution focal loss weight
```

### Data Augmentation (Medical-Optimized)
```yaml
# Conservative augmentation for medical data
hsv_h: 0.0     # No hue changes for medical images
hsv_s: 0.0     # No saturation changes for grayscale
hsv_v: 0.15    # Slight brightness variation
degrees: 3.0   # Small rotation for medical accuracy
translate: 0.1 # Small translation
scale: 0.15    # Scale variation
shear: 1.0     # Minimal shear
perspective: 0.0  # No perspective distortion
flipud: 0.5    # Vertical flip
fliplr: 0.5    # Horizontal flip
mosaic: 0.3    # Reduced mosaic for medical data
mixup: 0.0     # No mixup for medical detection
copy_paste: 0.0  # No copy-paste augmentation
```

## Checkpoint Management

### Automatic Cleanup Configuration
```python
# In v3_model.py
MAX_BEST_CHECKPOINTS = 3
MAX_LAST_CHECKPOINTS = 3  
MAX_OTHER_CHECKPOINTS = 2
CLEANUP_TRIGGER_THRESHOLD = 10  # Cleanup when > 10 total files
```

### Checkpoint Retention Policy
- **Best checkpoints**: Keep 3 best performing models based on validation metrics
- **Last checkpoints**: Keep 3 most recent checkpoints for resuming training
- **Other checkpoints**: Keep 2 additional important checkpoints (epoch milestones)
- **Total limit**: Maximum 8 checkpoint files maintained automatically

## Performance Tuning

### Batch Size Guidelines
| GPU Memory | Recommended Batch Size | Image Size | Expected Training Time (50 epochs) |
|------------|----------------------|------------|-----------------------------------|
| 4GB        | 1                    | 320px      | ~45 minutes                       |
| 6GB        | 2-3                  | 416px      | ~30 minutes                       |
| 8GB        | 4-6                  | 480px      | ~25 minutes                       |
| 12GB+      | 8-16                 | 640px      | ~15 minutes                       |

### Worker Configuration
```python
# CPU workers based on system
workers = min(4, os.cpu_count())  # Automatic detection
workers = 1  # For memory-constrained systems
workers = 2  # For balanced systems
workers = 4  # For high-performance systems
```

## Troubleshooting Common Issues

### CUDA Out of Memory
1. **Reduce batch size**: Start with batch=1, increase gradually
2. **Reduce image size**: Use 320px or 256px for very limited memory
3. **Disable AMP**: Set `amp: false` in configuration
4. **Clear GPU memory**: Run `python clear_gpu.py`
5. **Set environment variables**: Run `./optimize_gpu.sh`

### Slow Training
1. **Increase batch size**: If memory allows, increase for better GPU utilization
2. **Enable AMP**: Set `amp: true` for newer GPUs (8GB+)
3. **Increase workers**: Match number of CPU cores
4. **Use SSD storage**: Store dataset on SSD for faster I/O
5. **Enable caching**: Set `cache: true` if memory allows

### Low Detection Accuracy
1. **Increase image size**: Use higher resolution (480px, 640px)
2. **Increase epochs**: Train for more epochs (50-100)
3. **Reduce augmentation**: Lower augmentation parameters for medical data
4. **Check preprocessing**: Ensure appropriate preprocessing for image type
5. **Validate dataset**: Check annotation quality and format

### Memory Fragmentation
1. **Use expandable segments**: Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
2. **Restart training**: Clear GPU memory and restart
3. **Reduce model complexity**: Use smaller model variants
4. **Monitor memory**: Use `nvidia-smi -l 1` to track usage

## Model Selection Guide

### YOLO11 Variants
- **YOLO11n (nano)**: 1.9M parameters, fastest inference, lowest memory usage
- **YOLO11s (small)**: 9.4M parameters, balanced speed and accuracy
- **YOLO11m (medium)**: 20.1M parameters, higher accuracy, more memory
- **YOLO11l (large)**: 25.3M parameters, best accuracy, highest memory usage

### Recommended Configurations
- **RTX 3050 4GB**: YOLO11n only
- **RTX 4060 8GB**: YOLO11n or YOLO11s
- **RTX 4070+ 12GB**: YOLO11s, YOLO11m, or YOLO11l

## Production Deployment

### Server Configuration
```yaml
# Production optimized settings
val: true
save: true
save_period: -1  # Save only best and last
plots: false     # Disable plots to save memory
verbose: false   # Reduce logging overhead
deterministic: true  # Reproducible results
seed: 42         # Fixed random seed
```

### Inference Optimization
```python
# Model loading for inference
model = YOLO('path/to/best.pt')
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Batch inference for multiple images
results = model.predict(
    source='path/to/images',
    conf=0.25,
    iou=0.7,
    max_det=50,
    save=True
)
```

This configuration guide ensures optimal performance across different hardware configurations while maintaining medical imaging accuracy standards.
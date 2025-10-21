# NephroScan Data Storage & Management

This document explains the data organization and storage structure for the NephroScan kidney stone detection system.

## Directory Structure

### Input Data
```
data/                          # Main dataset directory
├── train/                     # Training data
│   ├── images/                # Training images
│   └── labels/                # Training labels (YOLO format)
├── valid/                     # Validation data
│   ├── images/                # Validation images
│   └── labels/                # Validation labels
├── test/                      # Test data
│   ├── images/                # Test images
│   └── labels/                # Test labels (optional)
└── data.yaml                  # Dataset configuration
```

### Output Results
```
output/                        # Generated results and outputs
├── detections/                # Individual detection results
│   └── detected_YYYYMMDD_HHMMSS.jpg
├── training_runs/             # Training session outputs
│   └── training_YYYYMMDD_HHMMSS/
│       ├── logs/              # Training logs
│       ├── plots/             # Visualization plots
│       ├── results/           # Metrics and configurations
│       └── reports/           # Generated analysis reports
└── exports/                   # Model exports and conversions
```

### Model Storage
```
models/                        # Model configurations and weights
└── yolov8_kidney_stone_v3/    # Main model directory
    ├── weights/               # Trained model weights
    │   ├── best.pt            # Best performing model
    │   └── last.pt            # Latest checkpoint
    ├── configs/               # Training configurations
    │   └── model_config_v3.yaml
    └── scripts/               # Training scripts
        └── train_v3.py
```

### Application Data
```
logs/                          # Application logs
├── app_YYYYMMDD.log          # Daily application logs
└── training_YYYYMMDD.log     # Training session logs
```

## Data Types and Operations

### 1. **Dataset Preparation**
- **Location**: `data/` directory
- **Format**: YOLO format with images and corresponding label files
- **Structure**: Separate train/valid/test splits
- **Configuration**: `data.yaml` defines paths and class information

### 2. **Single Image Detection**
- **Input**: Individual medical images (CT scans, X-rays, ultrasounds)
- **Output**: `output/detections/detected_YYYYMMDD_HHMMSS.jpg`
- **Features**: Annotated images with bounding boxes and confidence scores
- **Preprocessing**: Automatic adaptive preprocessing applied

### 3. **Model Training**
- **Output**: `output/training_runs/training_YYYYMMDD_HHMMSS/`
- **Components**:
  - Training logs and progress monitoring
  - Visualization plots (loss curves, confusion matrices)
  - Model checkpoints and final weights
  - Training configuration and arguments
- **Model Storage**: Best weights copied to `models/yolov8_kidney_stone_v3/weights/`

### 4. **Model Evaluation and Testing**
- **Metrics**: Precision, recall, mAP50, F1-score
- **Visualizations**: Confusion matrices, PR curves, F1 curves
- **Reports**: Comprehensive analysis reports with medical insights
- **Storage**: Organized within training run directories

## Data Formats

### Medical Images
- **Supported Formats**: JPEG, PNG, TIFF, DICOM
- **Preprocessing**: Automatic detection and enhancement
- **Resolution**: Adaptive handling from 320px to 640px+ 
- **Types**: CT scans, X-rays, ultrasounds, other medical imaging

### Labels (YOLO Format)
```
# Each line: class_id x_center y_center width height (normalized 0-1)
0 0.5 0.3 0.2 0.4
```

### Dataset Configuration (data.yaml)
```yaml
path: ./data
train: train/images
val: valid/images
test: test/images

nc: 1  # number of classes
names: ['kidney_stone']  # class names
```

## Storage Management

### Automatic Organization
- **Timestamped Files**: All outputs use YYYYMMDD_HHMMSS format
- **No Overwrites**: Unique timestamps prevent data loss
- **Structured Hierarchy**: Logical folder organization
- **Legacy Preservation**: Historical data maintained

### Disk Space Considerations
- **Training Outputs**: Can be large due to checkpoints and plots
- **Model Weights**: PyTorch .pt files typically 5-50MB
- **Detection Results**: Depends on image resolution and batch size
- **Logs**: Text files, minimal space usage

### Cleanup Recommendations
```bash
# Remove old detection results (older than 30 days)
find output/detections -name "*.jpg" -mtime +30 -delete

# Clean up old training runs (keep recent 5)
ls -t output/training_runs/ | tail -n +6 | xargs rm -rf

# Archive old logs
tar -czf logs_archive_$(date +%Y%m%d).tar.gz logs/*.log
```

## Configuration Management

Storage paths are managed through `src/utils/config.py`:

```python
from src.utils.config import Config

config = Config()

# Main directories
data_dir = config.data_dir              # data/
output_dir = config.output_dir          # output/
models_dir = config.models_dir          # models/

# Specific paths
detection_dir = config.detection_dir    # output/detections/
training_dir = config.training_dir      # output/training_runs/
logs_dir = config.logs_dir              # logs/
```

### Configuration Parameters
- **Data Paths**: Centralized dataset location management
- **Output Organization**: Automatic directory creation
- **Model Paths**: Dynamic model weight location handling
- **GPU Settings**: Hardware-specific optimizations

## File Naming Conventions

### Timestamp Format
- **Pattern**: `YYYYMMDD_HHMMSS`
- **Example**: `20251021_143022`
- **Benefits**: Chronological sorting, no conflicts

### File Types
```bash
# Detection results
detected_20251021_143022.jpg
detected_20251021_143022_confidence.json

# Training outputs
training_20251021_143022/
├── results.csv
├── confusion_matrix.png
├── training_progress.log
└── best_model.pt

# Application logs
app_20251021.log
training_20251021_143022.log
```

## Data Access Patterns

### Reading Data
```python
# Load configuration
config = Config()

# Access dataset
train_images = config.data_dir / "train" / "images"
train_labels = config.data_dir / "train" / "labels"

# Load model weights
model_path = config.models_dir / "yolov8_kidney_stone_v3" / "weights" / "best.pt"
```

### Writing Results
```python
# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = config.output_dir / "training_runs" / f"training_{timestamp}"
output_path.mkdir(parents=True, exist_ok=True)

# Save detection result
detection_path = config.output_dir / "detections" / f"detected_{timestamp}.jpg"
```

## Backup and Recovery

### Critical Data
- **Model Weights**: `models/yolov8_kidney_stone_v3/weights/`
- **Dataset**: `data/` directory
- **Configuration**: `src/utils/config.py`

### Backup Strategy
```bash
# Full backup
tar -czf nephroscan_backup_$(date +%Y%m%d).tar.gz \
    models/ data/ output/ docs/ src/

# Models only
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Dataset only
tar -czf dataset_backup_$(date +%Y%m%d).tar.gz data/
```

### Recovery
- Restore from backups to appropriate directories
- Verify configuration paths in `src/utils/config.py`
- Test model loading and data access
- Validate GUI functionality
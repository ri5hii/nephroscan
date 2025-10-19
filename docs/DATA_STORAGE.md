NephroScan Data Storage & Results Structure
==========================================

This document explains where data and results from GUI training and testing operations are stored.

##  Directory Structure

### Input Data
```
 data/                          # Main dataset directory
  train/                     # Training data
     images/                # Training images
     labels/                # Training labels
  valid/                     # Validation data
     images/                # Validation images
     labels/                # Validation labels
  test/                      # Test data
     images/                # Test images
     labels/                # Test labels
  data.yaml                  # Dataset configuration
```

### Output Results
```
 output/                        # All generated results
  detections/                # Individual detection results
     detected_YYYYMMDD_HHMMSS.jpg
  training_cycles/           #  Organized by training cycle
      training_YYYYMMDD_HHMMSS/  # Individual training cycle
          logs/              # Training session logs
             training_v3_YYYYMMDD_HHMMSS.log
          plots/             # All training visualizations
             BoxF1_curve.png
             confusion_matrix.png
             results.png
             training_batch*.jpg
          results/           # Training results & configs
             results.csv
             args.yaml
          reports/           # Generated analysis reports
              training_summary_YYYYMMDD_HHMMSS.html
```

### Model Storage
```
 models/                        # Trained models
  yolov8_kidney_stone_v3/    # Main model directory
      weights/               # All model weights (current + legacy backups)
         best_v3_20250920_170700.pt        # Current production model
         last_v3_20250920_170700.pt        # Current last checkpoint
         legacy_best_20250920_170700.pt    # Legacy model (migrated)
         legacy_last_20250920_170700.pt    # Legacy checkpoint (migrated)
      training_YYYYMMDD_HHMMSS/  #  Individual training runs (auto-created)
         weights/           # Training checkpoints
         results.png
         args.yaml
      configs/               # Training configurations
      scripts/               # Training scripts
```

**Important Notes:**
- **`weights/`** folder contains **all model weights** - current, legacy, and training backups
- **Legacy results migrated** to appropriate `output/` directories - no more `results/` folder
- **`training_YYYYMMDD_HHMMSS/`** folders contain **individual training runs** from GUI (auto-created by YOLO)
- **New training** automatically copies best model to `weights/` folder with timestamp
- **Clean separation** - models contain only models, results stored in `output/`
```
 models/                        # Trained models
  yolov8_kidney_stone_v3/    # Main model directory
      weights/               # All model weights (main + backups)
         best_v3_20250920_170700.pt    # Current production model
         best_v3_YYYYMMDD_HHMMSS.pt    # New training results
      results/               #  Legacy training results (preserved)
         training_v3_20250920_170659.log
         yolov8s_v3_20250920_170700/
             results.png
             confusion_matrix.png
             weights/       # Legacy model weights
      training_YYYYMMDD_HHMMSS/  #  Individual training runs (auto-created)
         weights/           # Training checkpoints
         results.png
         args.yaml
      configs/               # Training configurations
      scripts/               # Training scripts
```

**Important Notes:**
- **`weights/`** folder contains **all model weights** - current production model and training backups
- **`results/`** folder contains **legacy training data** from previous training sessions - **preserved**
- **`training_YYYYMMDD_HHMMSS/`** folders contain **individual training runs** from GUI (auto-created by YOLO)
- **New training** automatically copies best model to `weights/` folder with timestamp
- **No redundant** folder structure - clean and organized

##  Result Types by Operation

### 1. **Single Image Detection** (Test tab)
- **Storage**: `output/detections/`
- **Files**: `detected_YYYYMMDD_HHMMSS.jpg` - Annotated image with bounding boxes
- **Organization**: Individual files by detection timestamp

### 2. **Model Training** (Train tab)  
- **Storage**: `output/training_cycles/training_YYYYMMDD_HHMMSS/`
- **Organization**: All training-related files grouped by training cycle
- **Subfolders**:
  - `logs/` - Training progress logs
  - `plots/` - All training visualizations (curves, matrices, batches)
  - `results/` - Training metrics and configuration files
  - `reports/` - Generated analysis reports
- **Model weights**: Copied to `models/yolov8_kidney_stone_v3/weights/`

### 3. **Model Evaluation** (Analysis features)
- **Storage**: `output/training_cycles/training_YYYYMMDD_HHMMSS/results/`
- **Files**: Performance metrics, configuration files, CSV results
- **Plots**: Stored in corresponding `plots/` folder

### 4. **Analysis Reports** (Generated from GUI)
- **Storage**: `output/training_cycles/training_YYYYMMDD_HHMMSS/reports/`
- **Files**: Comprehensive analysis reports tied to specific training cycle

##  Configuration

The storage paths are managed by `src/nephroscan/utils/config.py`:

```python
# Output directories - organized by training cycle
self.output_dir = self.project_root / "output"

# Training cycle organization
config.get_training_cycle_dir(timestamp)      # output/training_cycles/training_YYYYMMDD_HHMMSS/
config.get_training_logs_dir(timestamp)       # output/training_cycles/training_YYYYMMDD_HHMMSS/logs/
config.get_training_plots_dir(timestamp)      # output/training_cycles/training_YYYYMMDD_HHMMSS/plots/
config.get_training_results_dir(timestamp)    # output/training_cycles/training_YYYYMMDD_HHMMSS/results/
config.get_training_reports_dir(timestamp)    # output/training_cycles/training_YYYYMMDD_HHMMSS/reports/

# Individual detection results
config.get_detection_output_dir()             # output/detections/
```

##  File Naming Convention

All result files use timestamp-based naming:
- Format: `operation_YYYYMMDD_HHMMSS.extension`
- Example: `detected_20250920_143022.jpg`
- This ensures no file overwrites and clear chronological ordering

##  Cleanup

Old results are retained for historical reference. To clean up:
1. Navigate to specific output subdirectory
2. Remove old files manually or use date-based scripts
3. Training checkpoints can be large - clean periodically

##  Access from Code

```python
from nephroscan.utils.config import config

# Get output paths
detection_dir = config.get_detection_output_dir()
training_dir = config.get_training_output_dir()
test_dir = config.get_test_output_dir()
```
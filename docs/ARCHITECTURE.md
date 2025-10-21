# NephroScan v3 - Project Architecture

## Current Project Structure

NephroScan is organized as a modular medical imaging system with clear separation of concerns.

### Directory Structure

```
nephroscan/
├── src/                     # Core application source code
│   ├── __init__.py
│   ├── main.py             # Application entry point
│   ├── backend/            # Machine learning backend
│   │   ├── __init__.py
│   │   └── v3_model.py     # YOLO11 model implementation
│   ├── gui/                # PyQt5 user interface
│   │   ├── __init__.py
│   │   ├── main_window.py  # Main GUI application
│   │   └── widgets/        # UI components
│   │       └── __init__.py
│   └── utils/              # Utilities and configuration
│       ├── __init__.py
│       ├── config.py       # Configuration management
│       ├── image_preprocessor.py  # Adaptive preprocessing
│       └── yolo_utils.py   # YOLO training utilities
├── scripts/                # Utility scripts
│   ├── install.py          # Installation script
│   ├── remove_emojis.py    # Emoji cleanup utility
│   └── sh/                 # Shell scripts
│       └── optimize_gpu.sh # GPU optimization
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md     # This file
│   ├── CONFIGURATION.md    # Configuration guide
│   ├── DATA_STORAGE.md     # Data management
│   ├── configuration.md    # Troubleshooting
│   ├── contributing.md     # Contributing guidelines
│   ├── features.md         # Feature documentation
│   ├── installation.md     # Setup instructions
│   ├── performance.md      # Performance metrics
│   ├── updates.md          # Version history
│   └── usage.md            # Usage guide
├── data/                   # Dataset storage
│   ├── train/              # Training data
│   ├── valid/              # Validation data
│   └── test/               # Test data
├── models/                 # Model configurations and weights
│   └── yolov8_kidney_stone_v3/
│       ├── configs/        # Training configurations
│       └── scripts/        # Training scripts
├── logs/                   # Application logs
├── output/                 # Generated outputs
├── .venv/                  # Virtual environment
├── launch.py               # Main application launcher
├── clear_gpu.py            # GPU memory cleanup
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
```

## Key Components

### Backend Module (`src/backend/`)
- **v3_model.py**: Core machine learning functionality
  - YOLO11 model implementation
  - Training, testing, and inference pipelines
  - Real-time progress monitoring
  - GPU memory optimization

### GUI Module (`src/gui/`)
- **main_window.py**: Main application interface
  - PyQt5-based graphical interface
  - Three-tab workflow (Upload/Analysis/Results)
  - Progress monitoring and visualization
  - User interaction handling
- **widgets/**: Reusable UI components
  - Custom widgets for medical imaging
  - Report generation components

### Utils Module (`src/utils/`)
- **config.py**: Configuration management
  - Centralized settings and paths
  - Training parameters
  - Hardware optimization settings
- **image_preprocessor.py**: Adaptive preprocessing system
  - CT scan detection and processing
  - Quality assessment algorithms
  - Medical image enhancement
- **yolo_utils.py**: YOLO-specific utilities
  - Training utilities and helpers
  - Dataset preparation functions

### Scripts (`scripts/`)
- **install.py**: Automated installation and setup
- **remove_emojis.py**: Codebase maintenance utility
- **sh/optimize_gpu.sh**: GPU memory optimization script

## Architecture Principles

### 1. **Modular Design**
- Clear separation between ML backend, GUI, and utilities
- Loosely coupled components for easy testing and maintenance
- Reusable modules across different workflows

### 2. **Adaptive Processing**
- Automatic detection of image types (CT scans, standard medical images)
- Dynamic preprocessing based on image characteristics
- Hardware-aware optimization for different GPU configurations

### 3. **Medical-First Approach**
- Conservative training parameters for medical accuracy
- Professional reporting and documentation features
- DICOM support and medical imaging standards compliance

### 4. **Memory Optimization**
- GPU memory management for low-end hardware (RTX 3050 4GB)
- Automatic batch size adaptation
- Checkpoint cleanup and storage optimization

## Usage

### Launching the Application
```bash
# From project root
python launch.py
```

### Installing Dependencies
```bash
# Enhanced installation script
python scripts/install.py
```

### GPU Optimization
```bash
# Optimize GPU memory before training
./scripts/sh/optimize_gpu.sh
```

### Importing Modules (for development)
```python
# Import main components
from src.backend.v3_model import V3ModelBackend
from src.gui.main_window import KidneyStoneDetectionGUI
from src.utils.config import Config

# Use configuration
config = Config()
dataset_path = config.data_dir
model_path = config.v3_model_path
```

## Technical Features

### Adaptive Preprocessing System
The system automatically detects image types and applies appropriate preprocessing:

- **CT Scan Processing**: High-resolution (≥640×640) with detail preservation
- **Low Quality Enhancement**: Noise reduction and adaptive histogram equalization
- **Standard Medical Processing**: Balanced preprocessing for general medical images

### Memory Management
- **Multi-GPU Support**: Optimized for RTX 3050 4GB and higher-end GPUs
- **Automatic Checkpoint Cleanup**: Maintains only essential checkpoints
- **Batch Size Adaptation**: Dynamic sizing based on available GPU memory
- **Real-time Monitoring**: GPU memory usage tracking during training

### Model Architecture
- **YOLO11 Nano**: Memory-efficient model optimized for medical imaging
- **Single Class Detection**: Specialized for kidney stone identification
- **Medical-Optimized Training**: Conservative learning rates for medical accuracy
- **Adaptive Configuration**: Automatically adjusts to available hardware

## Data Flow

1. **Input**: Medical images (CT scans, X-rays, ultrasounds)
2. **Preprocessing**: Adaptive enhancement based on image characteristics
3. **Detection**: YOLO11-based kidney stone detection
4. **Analysis**: Confidence scoring and bounding box generation
5. **Output**: Professional reports and visualizations

## Development Guidelines

### Adding New Features
1. Follow the modular structure
2. Add new modules to appropriate directories
3. Update configuration in `src/utils/config.py`
4. Document changes in relevant docs

### Testing
- Unit tests should be added to a `tests/` directory
- Test individual modules in isolation
- Validate on different GPU configurations

### Documentation
- Update relevant documentation files in `docs/`
- Follow the established markdown structure
- Include code examples and usage instructions
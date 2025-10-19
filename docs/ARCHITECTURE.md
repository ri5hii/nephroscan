# NephroScan v3 - Restructured Project Architecture

##  New Modular Structure

The NephroScan project has been completely restructured into a clean, modular architecture for better maintainability, scalability, and ease of reference.

###  Directory Structure

```
nephroscan/
  src/nephroscan/              # Core application source code
     backend/                 # Machine learning backend
       __init__.py
       v3_model.py            # V3ModelBackend class (renamed from v3_backend.py)
     gui/                    # PyQt5 user interface
       __init__.py
       main_window.py         # KidneyStoneDetectionGUI (renamed from main.py)
        widgets/            # Reusable GUI components
           __init__.py
           report_generator.py # PDFReportGenerator (renamed from report.py)
     utils/                  # Utilities and configuration
       __init__.py
       config.py             # Centralized configuration management
       yolo_utils.py         # YOLO training utilities
    __init__.py               # Package initialization
  scripts/                   # Installation and utility scripts
    install.py               # Enhanced installation script
  docs/                     # Documentation and guides
  tests/                    # Unit tests and validation
  data/                     # Dataset storage (unchanged)
  models/                   # Model weights and outputs (unchanged)
  .venv/                    # Virtual environment (unchanged)
 launch.py                    # Main application launcher (enhanced)
 requirements.txt             # Consolidated dependencies
 README.md                    # Updated project documentation
```

##  Key Changes

### 1. **Modular Package Structure**
- **Before**: All files in single directory with unclear organization
- **After**: Proper Python package structure with logical module separation

### 2. **Centralized Configuration**
- **New**: `src/nephroscan/utils/config.py` manages all paths and settings
- **Benefits**: Single source of truth for configuration, easy environment switching

### 3. **Enhanced Import System**
- **Before**: Relative imports and hardcoded paths
- **After**: Proper package imports and dynamic path resolution

### 4. **Cleaner File Names**
- `v3_backend.py` → `backend/v3_model.py`
- `main.py` → `gui/main_window.py`
- `report.py` → `gui/widgets/report_generator.py`

### 5. **Professional Launch System**
- **New**: Comprehensive launcher with dependency checking
- **New**: Enhanced installation script with better error handling

##  Module Responsibilities

### Backend Module (`src/nephroscan/backend/`)
- `v3_model.py`: Core ML functionality
  - V3ModelBackend class
  - Training, testing, inference
  - Real-time progress monitoring
  - Advanced metrics and visualization

### GUI Module (`src/nephroscan/gui/`)
- `main_window.py`: Main application interface
  - KidneyStoneDetectionGUI class
  - Three-tab interface (Upload/Analysis/Results)
  - Progress monitoring and user interaction
- `widgets/`: Reusable components
  - `report_generator.py`: PDF report generation

### Utils Module (`src/nephroscan/utils/`)
- `config.py`: Configuration management
  - Centralized path management
  - Training parameters
  - GUI settings
- `yolo_utils.py`: YOLO-specific utilities
  - YOLOTrainer class
  - Dataset preparation functions

##  Usage with New Structure

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

### Importing Modules (for development)
```python
# Import main components
from nephroscan.backend.v3_model import V3ModelBackend
from nephroscan.gui.main_window import KidneyStoneDetectionGUI
from nephroscan.utils.config import config

# Use centralized configuration
dataset_path = config.dataset_path
model_path = config.v3_model_path
```

##  Benefits of Restructuring

1. ** Clean Organization**: Clear separation of concerns
2. ** Easy Maintenance**: Logical module structure
3. ** Scalability**: Easy to add new features
4. ** Testability**: Isolated components for unit testing
5. ** Documentation**: Better code organization and documentation
6. ** Reusability**: Modular components can be reused
7. ** Configuration**: Centralized settings management

##  Import Path Changes

### Old Structure Imports
```python
# OLD - Direct file imports
from v3_backend import V3ModelBackend
from report import PDFReportGenerator
import yolov_utils
```

### New Structure Imports
```python
# NEW - Package-based imports
from nephroscan.backend.v3_model import V3ModelBackend
from nephroscan.gui.widgets.report_generator import PDFReportGenerator
from nephroscan.utils.yolo_utils import YOLOTrainer
from nephroscan.utils.config import config
```

##  Migration Summary

The restructuring maintains all functionality while providing:
-  **Clean modular architecture**
-  **Centralized configuration management**
-  **Professional package structure**
-  **Enhanced launcher and installation**
-  **Improved documentation**
-  **Better maintainability**

All original features remain intact:
- Real-time epoch progress monitoring
- Advanced visualization and metrics
- Comprehensive PDF reporting
- Three-tab GUI interface
- Full training/testing/inference pipeline

The restructured NephroScan v3 is now production-ready with professional-grade organization and maintainability.
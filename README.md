# NephroScan v3 - Advanced Kidney Stone Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://pypi.org/project/PyQt5/)
[![YOLO11](https://img.shields.io/badge/Model-YOLO11-orange.svg)](https://github.com/ultralytics/ultralytics)
[![CUDA](https://img.shields.io/badge/CUDA-Optimized-green.svg)](https://developer.nvidia.com/cuda-zone)

A comprehensive medical imaging system for kidney stone detection and analysis using advanced deep learning models with adaptive preprocessing and memory optimization for various GPU configurations.

## ✨ Key Features

### 🔬 **Adaptive Preprocessing System**
- **CT Scan Detection**: Automatically detects and preserves detail in high-resolution CT scans (640×640+ resolution)
- **Quality Assessment**: Intelligent image quality evaluation with appropriate enhancement
- **Medical Image Processing**: Specialized preprocessing for different medical imaging modalities
- **Coordinate Scaling**: Maintains accurate detection mapping across different image resolutions

### 🧠 **AI Model Architecture**
- **YOLO11 Nano**: Memory-efficient model optimized for medical imaging
- **Single Class Detection**: Specialized for kidney stone identification
- **Medical-Optimized Training**: Conservative learning rates and loss weights tuned for medical accuracy
- **Adaptive Configuration**: Automatically adjusts to available GPU memory

### 💾 **Memory Management & GPU Optimization**
- **Multi-GPU Support**: Optimized for RTX 3050 4GB and higher-end GPUs
- **Automatic Checkpoint Cleanup**: Maintains only essential checkpoints (3 best + 3 last + 2 other)
- **Batch Size Adaptation**: Dynamic batch sizing based on available GPU memory
- **Memory Fragmentation Prevention**: Environment variable optimization for CUDA
- **Real-time Monitoring**: GPU memory usage tracking during training

### �️ **User Interface & Workflow**
- **PyQt5 GUI**: Intuitive graphical interface for medical professionals
- **Three-Tab Workflow**: Training, Testing, and Inference in organized tabs
- **Real-time Processing**: Live image preprocessing and detection visualization
- **Progress Monitoring**: Detailed training progress with memory usage tracking
- **Report Generation**: Professional PDF report generation for medical documentation

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+ 
- CUDA-compatible GPU (minimum: 3.68 GB VRAM, recommended: 8GB+)
- NVIDIA drivers and CUDA toolkit

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd nephroscan

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up GPU optimization (for RTX 3050/4GB GPUs)
chmod +x optimize_gpu.sh
./optimize_gpu.sh

# Launch the application
python src/main.py
```

## 💻 Hardware Configuration

### GPU Memory Optimization

The system automatically detects your hardware and optimizes settings:

**RTX 3050 4GB (or similar low-memory GPUs):**
- Image size: 320px
- Batch size: 1
- Workers: 1
- AMP: Disabled
- Cache: Disabled

**RTX 4060/3060 8GB:**
- Image size: 480px
- Batch size: 4-6
- Workers: 2
- AMP: Optional

**RTX 4070+ 12GB+:**
- Image size: 640px
- Batch size: 8-16
- Workers: 4
- AMP: Enabled

### Memory Optimization Script

Run before training to optimize GPU memory:
```bash
./optimize_gpu.sh
```

This automatically sets:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `CUDA_LAUNCH_BLOCKING=1`
- `CUDA_VISIBLE_DEVICES=0`

## 📁 Project Structure

```
nephroscan/
├── src/                          # Source code
│   ├── gui/                      # PyQt5 interface
│   │   ├── main_window.py        # Main application window
│   │   └── widgets/              # UI components
│   ├── backend/                  # Core functionality
│   │   └── v3_model.py           # YOLO11 model backend with adaptive preprocessing
│   └── utils/                    # Utilities
│       ├── config.py             # Configuration management
│       └── image_preprocessor.py # Adaptive preprocessing system
├── models/                       # Model configurations
│   └── yolov8_kidney_stone_v3/
│       ├── configs/
│       │   └── model_config_v3.yaml  # GPU-optimized training configuration
│       └── weights/              # Trained model checkpoints
├── data/                         # Dataset structure (YOLO format)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   └── test/
├── docs/                         # Documentation
├── scripts/                      # Utility scripts
├── optimize_gpu.sh               # GPU memory optimization script
└── clear_gpu.py                  # GPU memory cleanup utility
```

## � Adaptive Preprocessing System

The system automatically detects image types and applies appropriate preprocessing:

### CT Scan Processing
**Detection criteria**: Resolution ≥ 640×640, high contrast ratio
- **Processing**: Detail preservation with minimal smoothing
- **Enhancement**: Gentle contrast adjustment maintaining medical accuracy
- **Use case**: High-resolution CT scans requiring detail preservation

### Low Quality Enhancement  
**Detection criteria**: Low contrast, noise presence, resolution < 640×640
- **Processing**: Noise reduction with edge preservation
- **Enhancement**: Adaptive histogram equalization
- **Use case**: Lower quality images requiring enhancement

### Standard Medical Processing
**Default processing**: Balanced preprocessing for general medical images
- **Processing**: Standard normalization and resizing
- **Enhancement**: Medical-optimized contrast and brightness adjustment
- **Use case**: Standard medical imaging modalities

## 🏃‍♂️ Usage Guide

### Training Workflow
1. **Prepare Dataset**: Organize images and labels in YOLO format in `data/` directory
2. **Optimize GPU**: Run `./optimize_gpu.sh` for memory optimization
3. **Launch GUI**: `python src/main.py`
4. **Configure Training**: Set epochs, batch size based on your GPU
5. **Start Training**: Monitor progress and memory usage in real-time
6. **Checkpoint Management**: Automatic cleanup maintains optimal storage

### Inference Workflow
1. **Load Model**: Select trained checkpoint through GUI
2. **Input Selection**: Choose single images or batch processing
3. **Preprocessing**: Automatic adaptive preprocessing applied
4. **Detection**: Real-time kidney stone detection with confidence scores
5. **Results**: Bounding boxes, confidence metrics, and medical analysis
6. **Report Generation**: Professional PDF reports for documentation

## 🔧 Configuration & Troubleshooting

### CUDA Out of Memory Solutions
```bash
# 1. Run GPU optimization
./optimize_gpu.sh

# 2. Clear GPU memory
python clear_gpu.py

# 3. Monitor memory usage
nvidia-smi -l 1

# 4. Reduce configuration in model_config_v3.yaml:
# - batch: 1 (minimum)
# - imgsz: 320 (or 256 for extreme cases)
# - workers: 1
# - cache: false
```

### Performance Optimization
- **SSD Storage**: Use SSD for datasets to improve I/O
- **CPU Workers**: Increase `workers` based on CPU cores
- **AMP Training**: Enable for newer GPUs (8GB+) to improve speed
- **Batch Size**: Increase gradually based on available memory

### Training Configuration
Edit `models/yolov8_kidney_stone_v3/configs/model_config_v3.yaml`:
```yaml
# For RTX 3050 4GB
epochs: 25
imgsz: 320
batch: 1
workers: 1
amp: false

# For RTX 4060 8GB  
epochs: 50
imgsz: 480
batch: 4
workers: 2
amp: false

# For RTX 4070+ 12GB+
epochs: 100
imgsz: 640
batch: 8
workers: 4
amp: true
```

## 📊 Model Performance & Metrics

### Current Performance (YOLO11 Nano)
- **mAP50**: Optimized for medical accuracy
- **Precision**: High precision for clinical use
- **Recall**: Sensitive kidney stone detection
- **Training Speed**: ~45 minutes on RTX 3050 4GB
- **Inference Speed**: Real-time detection

### Memory Usage Benchmarks
| GPU Model | Max Batch Size | Optimal Image Size | Training Time (50 epochs) |
|-----------|---------------|-------------------|---------------------------|
| RTX 3050 4GB | 1 | 320px | ~45 minutes |
| RTX 4060 8GB | 4-6 | 480px | ~25 minutes |
| RTX 4070 12GB | 8-12 | 640px | ~15 minutes |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-adaptive-preprocessing`
3. Commit changes: `git commit -am 'Add adaptive preprocessing system'`
4. Push to branch: `git push origin feature-adaptive-preprocessing`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics YOLO11**: For the base model architecture and training framework
- **Medical Imaging Community**: For preprocessing insights and validation standards
- **PyQt5**: For the intuitive cross-platform graphical interface
- **CUDA Community**: For GPU optimization techniques and memory management

## 📞 Support & Issues

For issues, questions, or contributions:
- **GitHub Issues**: Create an issue for bugs or feature requests
- **Documentation**: Check the troubleshooting section above
- **Configuration Help**: Review GPU optimization guide
- **Performance Issues**: Monitor GPU memory with `nvidia-smi`

## 🔍 Recent Updates

### v3.2 - Memory Optimization & Adaptive Preprocessing
- ✅ Added adaptive preprocessing for CT scans, low-quality images, and standard medical images
- ✅ Implemented automatic GPU memory optimization for RTX 3050 4GB and similar GPUs
- ✅ Added automatic checkpoint management system
- ✅ Enhanced CUDA memory management with environment variable optimization
- ✅ Improved coordinate scaling for accurate detection mapping

---

**NephroScan v3** - Advancing kidney stone detection through AI, adaptive medical image processing, and intelligent memory management.
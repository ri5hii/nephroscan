# Installation & Setup

## Prerequisites
- Python 3.8+
- CUDA-compatible GPU (minimum: 3.68 GB VRAM, recommended: 8GB+)
- NVIDIA drivers and CUDA toolkit

## Quick Start
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
chmod +x scripts/sh/optimize_gpu.sh
./scripts/sh/optimize_gpu.sh

# Launch the application
python src/main.py
```

## Hardware Configuration

### GPU Memory Optimization
The system automatically detects your hardware and optimizes settings. See docs/features.md for details.

### Memory Optimization Script
Run before training to optimize GPU memory:
```bash
./scripts/sh/optimize_gpu.sh
```
This automatically sets:
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
- CUDA_LAUNCH_BLOCKING=1
- CUDA_VISIBLE_DEVICES=0

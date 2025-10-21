# Key Features

## Adaptive Preprocessing System
- CT Scan Detection: Automatically detects and preserves detail in high-resolution CT scans (640×640+ resolution)
- Quality Assessment: Intelligent image quality evaluation with appropriate enhancement
- Medical Image Processing: Specialized preprocessing for different medical imaging modalities
- Coordinate Scaling: Maintains accurate detection mapping across different image resolutions

## AI Model Architecture
- YOLO11 Nano: Memory-efficient model optimized for medical imaging
- Single Class Detection: Specialized for kidney stone identification
- Medical-Optimized Training: Conservative learning rates and loss weights tuned for medical accuracy
- Adaptive Configuration: Automatically adjusts to available GPU memory

## Memory Management & GPU Optimization
- Multi-GPU Support: Optimized for RTX 3050 4GB and higher-end GPUs
- Automatic Checkpoint Cleanup: Maintains only essential checkpoints (3 best + 3 last + 2 other)
- Batch Size Adaptation: Dynamic batch sizing based on available GPU memory
- Memory Fragmentation Prevention: Environment variable optimization for CUDA
- Real-time Monitoring: GPU memory usage tracking during training

## User Interface & Workflow
- PyQt5 GUI: Intuitive graphical interface for medical professionals
- Three-Tab Workflow: Training, Testing, and Inference in organized tabs
- Real-time Processing: Live image preprocessing and detection visualization
- Progress Monitoring: Detailed training progress with memory usage tracking
- Report Generation: Professional PDF report generation for medical documentation

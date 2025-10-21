#!/bin/bash
# GPU Memory Optimization Script for RTX 3050 4GB
# Sets environment variables for optimal CUDA memory management

echo " Setting up GPU memory optimization for RTX 3050..."

# CUDA memory allocator settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export TORCH_CUDA_ARCH_LIST=8.6

# Reduce memory fragmentation
export CUDA_VISIBLE_DEVICES=0  # Use only RTX 3050

echo " Environment variables set:"
echo "   PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "   CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Clear GPU memory
python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print(' GPU memory cleared')
    
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f' RTX 3050 ready: {total:.2f} GB available')
else:
    print(' CUDA not available')
"

echo " GPU optimization complete! Ready for training."

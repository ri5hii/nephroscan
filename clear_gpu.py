#!/usr/bin/env python3
"""
GPU Memory Cleaner for NephroScan
================================

Script to clear GPU memory before training.
"""

import gc
import torch

def clear_gpu_memory():
    """Clear GPU memory cache"""
    try:
        if torch.cuda.is_available():
            print("Clearing GPU memory...")
            
            # Empty cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Get memory info
            if torch.cuda.device_count() > 0:
                device = torch.cuda.current_device()
                total_memory = torch.cuda.get_device_properties(device).total_memory
                allocated = torch.cuda.memory_allocated(device)
                cached = torch.cuda.memory_reserved(device)
                
                print(f"GPU {device}:")
                print(f"  Total memory: {total_memory / 1024**3:.2f} GB")
                print(f"  Allocated: {allocated / 1024**3:.2f} GB")
                print(f"  Cached: {cached / 1024**3:.2f} GB")
                print(f"  Free: {(total_memory - allocated) / 1024**3:.2f} GB")
            
            print("GPU memory cleared successfully!")
            return True
        else:
            print("CUDA not available")
            return False
    except Exception as e:
        print(f"Error clearing GPU memory: {e}")
        return False

if __name__ == "__main__":
    clear_gpu_memory()
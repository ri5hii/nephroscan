#!/usr/bin/env python3
"""
YOLOv8 Kidney Stone Detection Training Script - Version 3
Enhanced model using comprehensive 1300+ image dataset
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import torch
import yaml

def setup_logging(log_dir):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_v3_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def check_environment():
    """Check if the environment is properly set up"""
    logger = logging.getLogger(__name__)
    
    # Check PyTorch and CUDA
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # Adjust batch size based on GPU memory
        if gpu_memory < 6:
            logger.warning("GPU memory < 6GB. Consider reducing batch size.")
    
    # Check Ultralytics
    try:
        from ultralytics import YOLO
        logger.info("Ultralytics YOLOv8 is available")
    except ImportError:
        logger.error("Ultralytics not found. Please install: pip install ultralytics")
        return False
    
    return True

def validate_dataset(dataset_path):
    """Validate dataset structure and files"""
    logger = logging.getLogger(__name__)
    dataset_path = Path(dataset_path)
    
    # Check required files and directories
    required_paths = [
        dataset_path / "train" / "images",
        dataset_path / "train" / "labels",
        dataset_path / "valid" / "images", 
        dataset_path / "valid" / "labels",
        dataset_path / "test" / "images",
        dataset_path / "test" / "labels",
    ]
    
    for path in required_paths:
        if not path.exists():
            logger.error(f"Required path not found: {path}")
            return False
        logger.info(f"✓ Found: {path}")
    
    # Count images and labels in each split
    splits = ['train', 'valid', 'test']
    total_images = 0
    
    for split in splits:
        images = list((dataset_path / split / "images").glob("*.jpg"))
        labels = list((dataset_path / split / "labels").glob("*.txt"))
        total_images += len(images)
        
        logger.info(f"{split.capitalize()} set: {len(images)} images, {len(labels)} labels")
        
        if len(images) != len(labels):
            logger.warning(f"Mismatch in {split}: {len(images)} images vs {len(labels)} labels")
    
    logger.info(f"Total dataset size: {total_images} images")
    return True

def create_model_config(model_dir, dataset_path, model_params):
    """Create model configuration file"""
    config = {
        "model_info": {
            "name": "YOLOv8 Kidney Stone Detection",
            "version": "3.0",
            "description": "Enhanced YOLOv8 model trained on comprehensive 1300+ image dataset",
            "created": datetime.now().isoformat(),
            "framework": "YOLOv8 (Ultralytics)",
        },
        "dataset": {
            "path": str(dataset_path),
            "config_file": str(model_dir / "configs" / "data_v3.yaml"),
            "classes": ["KidneyStone"],
            "num_classes": 1,
            "total_images": 1300,
            "splits": {
                "train": 1054,
                "valid": 123, 
                "test": 123
            }
        },
        "training_params": model_params,
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None
        }
    }
    
    config_path = Path(model_dir) / "configs" / "model_config_v3.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    return config_path

def train_model(dataset_path, model_dir, **kwargs):
    """Train YOLOv8 model"""
    logger = logging.getLogger(__name__)
    
    try:
        from ultralytics import YOLO
        
        # Model parameters - optimized for larger dataset
        model_size = kwargs.get('model_size', 'm')  # Use medium model for better performance
        epochs = kwargs.get('epochs', 100)
        batch_size = kwargs.get('batch_size', 16)   # Larger batch size for better training
        img_size = kwargs.get('img_size', 640)      # Standard YOLO size
        patience = kwargs.get('patience', 15)       # More patience for larger dataset
        
        # Adjust batch size based on GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        if gpu_memory < 6:
            batch_size = min(batch_size, 8)
            logger.info(f"Reduced batch size to {batch_size} due to GPU memory limitations")
        
        logger.info(f"Training parameters:")
        logger.info(f"  Model size: {model_size}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Image size: {img_size}")
        logger.info(f"  Patience: {patience}")
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"yolov8{model_size}_v3_{timestamp}"
        output_dir = Path(model_dir) / "results" / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        model_name = f"yolov8{model_size}.pt"
        logger.info(f"Loading {model_name}")
        model = YOLO(model_name)
        
        # Use the dataset config
        data_config = Path(model_dir) / "configs" / "data_v3.yaml"
        
        # Enhanced training arguments for larger dataset
        train_args = {
            'data': str(data_config),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'patience': patience,
            'project': str(Path(model_dir) / "results"),
            'name': run_name,
            'save': True,
            'save_period': 10,  # Save every 10 epochs
            'cache': True,      # Enable caching for faster training
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'workers': 4,       # More workers for larger dataset
            'verbose': True,
            'seed': 42,
            'exist_ok': True,
            
            # Optimized hyperparameters for medical imaging
            'lr0': 0.01,        # Higher learning rate for larger dataset
            'lrf': 0.01,        # Learning rate final
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,         # Box loss gain
            'cls': 0.5,         # Class loss gain  
            'dfl': 1.5,         # DFL loss gain
            'pose': 12.0,       # Pose loss gain
            'kobj': 1.0,        # Keypoint obj loss gain
            'label_smoothing': 0.0,
            'nbs': 64,          # Nominal batch size
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            
            # Data augmentation for medical images
            'mosaic': 0.5,      # Mosaic augmentation probability
            'mixup': 0.0,       # Mixup disabled for medical data
            'copy_paste': 0.0,  # Copy-paste disabled
            'degrees': 10.0,    # Image rotation degrees
            'translate': 0.1,   # Translation fraction
            'scale': 0.3,       # Image scale gain
            'shear': 3.0,       # Shear degrees
            'perspective': 0.0, # Perspective disabled for medical
            'flipud': 0.0,      # Vertical flip disabled
            'fliplr': 0.5,      # Horizontal flip probability
            'hsv_h': 0.02,      # Hue augmentation fraction
            'hsv_s': 0.4,       # Saturation augmentation fraction
            'hsv_v': 0.3,       # Value augmentation fraction
            'auto_augment': 'randaugment',
            'erasing': 0.4,     # Random erasing probability
        }
        
        logger.info("Starting training...")
        logger.info(f"Output directory: {output_dir}")
        
        # Train the model
        results = model.train(**train_args)
        
        # Save final weights to the weights directory
        best_weights = Path(model_dir) / "results" / run_name / "weights" / "best.pt"
        last_weights = Path(model_dir) / "results" / run_name / "weights" / "last.pt"
        
        if best_weights.exists():
            # Copy to main weights directory
            import shutil
            shutil.copy2(best_weights, Path(model_dir) / "weights" / f"best_v3_{timestamp}.pt")
            shutil.copy2(last_weights, Path(model_dir) / "weights" / f"last_v3_{timestamp}.pt")
            logger.info(f"Weights saved to: {Path(model_dir) / 'weights'}")
        
        # Save training summary
        summary = {
            "training_completed": datetime.now().isoformat(),
            "run_name": run_name,
            "model_size": model_size,
            "dataset_size": {
                "total_images": 1300,
                "train": 1054,
                "valid": 123,
                "test": 123
            },
            "final_weights": {
                "best": str(best_weights) if best_weights.exists() else None,
                "last": str(last_weights) if last_weights.exists() else None
            },
            "training_args": train_args,
            "results_directory": str(output_dir)
        }
        
        summary_path = Path(model_dir) / "configs" / f"training_summary_v3_{timestamp}.yaml"
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)
        
        logger.info("Training completed successfully!")
        logger.info(f"Training summary saved to: {summary_path}")
        
        return results, output_dir
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for kidney stone detection - Version 3')
    parser.add_argument('--model-size', choices=['n', 's', 'm', 'l', 'x'], default='m',
                        help='YOLOv8 model size (nano, small, medium, large, xlarge)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent  # Go up to nephroscan root
    model_dir = script_dir.parent
    dataset_path = project_root / "data"  # Use the new data folder
    
    # Setup logging
    logger = setup_logging(model_dir / "results")
    
    logger.info("="*60)
    logger.info("YOLOv8 Kidney Stone Detection Training - Version 3")
    logger.info("Enhanced Dataset with 1300+ Images")
    logger.info("="*60)
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Dataset path: {dataset_path}")
    
    # Check environment
    if not check_environment():
        logger.error("Environment check failed")
        return 1
    
    # Validate dataset
    if not validate_dataset(dataset_path):
        logger.error("Dataset validation failed")
        return 1
    
    # Create model configuration
    model_params = {
        'model_size': args.model_size,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'patience': args.patience
    }
    
    config_path = create_model_config(model_dir, dataset_path, model_params)
    logger.info(f"Model configuration saved to: {config_path}")
    
    # Train model
    try:
        results, output_dir = train_model(
            dataset_path=dataset_path,
            model_dir=model_dir,
            **model_params
        )
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Results available in: {output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
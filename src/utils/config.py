"""
Configuration module for NephroScan
===================================

Centralized configuration management for paths, model settings, and application parameters.
Loads configuration from YAML file for better maintainability.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """Central configuration class for NephroScan application"""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration from YAML file.
        
        Args:
            config_path: Optional path to config file. Defaults to model_config_v3.yaml
        """
        # Get project root directory (assuming config is in src/utils/)
        self.project_root = Path(__file__).parent.parent.parent
        
        # Load configuration from YAML file
        if config_path is None:
            config_path = self.project_root / "models" / "yolov8_kidney_stone_v3" / "configs" / "model_config_v3.yaml"
        
        self.config_data = self._load_config(config_path)
        
        # Initialize paths and settings from YAML
        self._init_paths()
        self._init_training_config()
        self._init_gui_config()
        
        # Create necessary directories
        self.create_directories()
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"⚠️  Config file not found at {config_path}. Using default values.")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"❌ Error parsing YAML config: {e}. Using default values.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if YAML file is not available"""
        return {
            "model": "models/pretrained/yolo11n.pt",  # Use existing YOLO11 nano model
            "data": "data/data.yaml",
            "epochs": 150,
            "batch": 16,
            "lr0": 0.01,
            "imgsz": 640,
            "patience": 50,
            "device": "auto",
            "workers": 8
        }
    
    def _init_paths(self):
        """Initialize all paths from configuration"""
        # Core directories (hardcoded since training config is focused on training params)
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"
        self.output_dir = self.project_root / "output"
        
        # Model paths
        model_path = self.config_data.get("model", "models/pretrained/yolo11n.pt")
        if model_path.startswith("models/"):
            self.yolo_pretrained = self.project_root / model_path
        else:
            self.yolo_pretrained = model_path
            
        model_name = "yolov8_kidney_stone_v3"
        self.v3_model_path = self.models_dir / model_name / "weights" / "best_v3_20250920_170700.pt"
        self.v3_legacy_path = self.models_dir / model_name / "weights" / "legacy_best_20250920_170700.pt"
        
        # Dataset configuration
        data_path = self.config_data.get("data", "data/data.yaml")
        if data_path.startswith("/"):
            # Absolute path
            self.dataset_path = Path(data_path).parent
        else:
            # Relative path
            self.dataset_path = self.project_root / Path(data_path).parent
        
        self.train_images = self.dataset_path / "train" / "images"
        self.train_labels = self.dataset_path / "train" / "labels"
        self.valid_images = self.dataset_path / "valid" / "images"
        self.valid_labels = self.dataset_path / "valid" / "labels"
        self.test_images = self.dataset_path / "test" / "images"
        self.test_labels = self.dataset_path / "test" / "labels"
    
    def _init_training_config(self):
        """Initialize training configuration from YAML"""
        # The config data is already flat (no nested 'training' section)
        # Convert YAML configuration to the format expected by the training system
        self.training_config = {
            # Model specification - use the absolute path to organized pretrained model
            "model": str(self.yolo_pretrained),
            
            # Basic training settings
            "epochs": self.config_data.get("epochs", 150),
            "batch": self.config_data.get("batch", 16),  # Fixed: batch_size -> batch
            "lr0": self.config_data.get("lr0", 0.01),    # Fixed: learning_rate -> lr0
            "imgsz": self.config_data.get("imgsz", 640), # Fixed: img_size -> imgsz
            "patience": self.config_data.get("patience", 50),
            "save_period": self.config_data.get("save_period", -1),
            
            # Advanced learning rate configuration
            "lrf": self.config_data.get("lrf", 0.01),
            "momentum": self.config_data.get("momentum", 0.937),
            "weight_decay": self.config_data.get("weight_decay", 0.0005),
            
            # Warmup configuration
            "warmup_epochs": self.config_data.get("warmup_epochs", 3),
            "warmup_momentum": self.config_data.get("warmup_momentum", 0.8),
            "warmup_bias_lr": self.config_data.get("warmup_bias_lr", 0.1),
            
            # Loss function weights
            "box": self.config_data.get("box", 7.5),
            "cls": self.config_data.get("cls", 0.5),
            "dfl": self.config_data.get("dfl", 1.5),
            
            # Data augmentation
            "hsv_h": self.config_data.get("hsv_h", 0.0),
            "hsv_s": self.config_data.get("hsv_s", 0.0),
            "hsv_v": self.config_data.get("hsv_v", 0.15),
            "degrees": self.config_data.get("degrees", 3.0),
            "translate": self.config_data.get("translate", 0.1),
            "scale": self.config_data.get("scale", 0.15),
            "shear": self.config_data.get("shear", 1.0),
            "perspective": self.config_data.get("perspective", 0.0),
            "flipud": self.config_data.get("flipud", 0.5),
            "fliplr": self.config_data.get("fliplr", 0.5),
            "mosaic": self.config_data.get("mosaic", 0.3),
            "mixup": self.config_data.get("mixup", 0.0),
            "copy_paste": self.config_data.get("copy_paste", 0.0),
            "close_mosaic": self.config_data.get("close_mosaic", 10),
            
            # Medical-specific settings
            "optimizer": self.config_data.get("optimizer", "auto"),
            "single_cls": self.config_data.get("single_cls", True),
            "amp": self.config_data.get("amp", True),
            "deterministic": self.config_data.get("deterministic", True),
            "seed": self.config_data.get("seed", 42),
            
            # Validation and detection settings
            "val": self.config_data.get("val", True),
            "plots": self.config_data.get("plots", True),
            "verbose": self.config_data.get("verbose", True),
            "rect": self.config_data.get("rect", False),
            "cos_lr": self.config_data.get("cos_lr", False),
            "multi_scale": self.config_data.get("multi_scale", False),
            "fraction": self.config_data.get("fraction", 1.0),
            
            # Detection thresholds
            "conf": self.config_data.get("conf", 0.25),
            "iou": self.config_data.get("iou", 0.7),
            "max_det": self.config_data.get("max_det", 300),
            "agnostic_nms": self.config_data.get("agnostic_nms", False),
            
            # Hardware configuration
            "device": self.config_data.get("device", "auto"),
            "workers": self.config_data.get("workers", 8),
            
            # Project configuration
            "project": self.config_data.get("project", "kidney_stone_detection"),
            "name": self.config_data.get("name", "yolov8s_medical_v1"),
            "exist_ok": self.config_data.get("exist_ok", True),
            
            # Additional settings
            "resume": self.config_data.get("resume", False),
            "profile": self.config_data.get("profile", False),
            "freeze": self.config_data.get("freeze", None),
            "retina_masks": self.config_data.get("retina_masks", False),
        }
    
    def _init_gui_config(self):
        """Initialize GUI configuration from YAML"""
        # GUI config not in training config, use defaults
        self.gui_config = {
            "window_title": "NephroScan v3 - Kidney Stone Detection System",
            "window_size": (1400, 900),
            "theme": "default"
        }
    
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.logs_dir,
            self.output_dir,
            self.output_dir / "temp_image_vault",     # Temporary storage for uploaded images
            self.output_dir / "inference_results",    # Single inference results
            self.output_dir / "training_cycles",      # Training cycle organized results
            self.models_dir / "yolov8_kidney_stone_v3" / "weights",  # Main model weights
            # Note: Training cycle subdirectories created dynamically per training session
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_dataset_yaml_path(self) -> Path:
        """Get the path to the dataset YAML configuration file"""
        return self.dataset_path / "data.yaml"
    
    def get_training_config_dict(self) -> dict:
        """Get complete training configuration for YOLO training"""
        # Force training outputs to go to organized training cycles directory
        project_path = self.validate_training_output_location(str(self.output_dir / "training_cycles"))
        
        return {
            **self.training_config,
            "data": str(self.get_dataset_yaml_path()),
            "project": project_path,  # Always use validated organized output
            "name": f"training_{self._get_timestamp()}",          # Timestamped name
            "exist_ok": True,  # Allow existing directories
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for training cycle naming"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def validate_training_output_location(self, output_path: str) -> str:
        """Validate and correct training output location to prevent misplacement"""
        output_path = Path(output_path)
        
        # If output is trying to go to models directory, redirect to training_cycles
        if str(output_path).find("models") != -1:
            self.log(f"⚠️  Redirecting training output from models directory to organized location")
            return str(self.output_dir / "training_cycles")
        
        # Ensure it goes to our organized training_cycles directory
        if not str(output_path).endswith("training_cycles"):
            return str(self.output_dir / "training_cycles")
        
        return str(output_path)
    
    def log(self, message: str):
        """Simple logging method"""
        print(f"[Config] {message}")
    
    def get_model_output_dir(self) -> Path:
        """Get the directory for final model weights (not training outputs)"""
        return self.models_dir / "yolov8_kidney_stone_v3" / "weights"
    
    def get_inference_results_dir(self) -> Path:
        """Get the directory for inference results"""
        return self.output_dir / "inference_results"
    
    def get_training_cycle_dir(self, timestamp: str = None) -> Path:
        """Get the directory for a specific training cycle"""
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / "training_cycles" / f"training_{timestamp}"
    
    def get_training_logs_dir(self, timestamp: str = None) -> Path:
        """Get the directory for training logs for a specific cycle"""
        return self.get_training_cycle_dir(timestamp) / "logs"
    
    def get_training_plots_dir(self, timestamp: str = None) -> Path:
        """Get the directory for training plots for a specific cycle"""
        return self.get_training_cycle_dir(timestamp) / "plots"
    
    def get_training_results_dir(self, timestamp: str = None) -> Path:
        """Get the directory for training results for a specific cycle"""
        return self.get_training_cycle_dir(timestamp) / "results"
    
    def get_training_reports_dir(self, timestamp: str = None) -> Path:
        """Get the directory for training reports for a specific cycle"""
        return self.get_training_cycle_dir(timestamp) / "reports"
    
    # Configuration accessors for YAML values
    def get_app_info(self) -> Dict[str, Any]:
        """Get application information from config"""
        return {
            "name": "NephroScan",
            "version": "3.0.0",
            "description": "Kidney Stone Detection System"
        }
    
    def get_medical_settings(self) -> Dict[str, Any]:
        """Get medical-specific settings from config"""
        return {
            "no_color_distortion": self.config_data.get("hsv_h", 0.0) == 0.0 and self.config_data.get("hsv_s", 0.0) == 0.0,
            "preserve_orientation": self.config_data.get("flipud", 0.5) == 0.5,  # Note: temp config has 0.5, not 0.0
            "normalize_intensity": True,
            "minimal_geometric_transform": True,
            "min_object_size": 10,
            "max_objects_per_image": 20
        }
    
    def get_performance_settings(self) -> Dict[str, Any]:
        """Get performance configuration from config"""
        return {
            "pin_memory": True,
            "num_workers": self.config_data.get("workers", 8),
            "prefetch_factor": 2,
            "mixed_precision": self.config_data.get("amp", True),
            "compile_model": False
        }
    
    def get_validation_settings(self) -> Dict[str, Any]:
        """Get validation configuration from config"""
        return {
            "val_freq": 1,
            "primary_metric": "mAP50-95",
            "save_best_metric": "mAP50",
            "conf_threshold": self.config_data.get("conf", 0.25),
            "iou_threshold": self.config_data.get("iou", 0.7),
            "max_detections": self.config_data.get("max_det", 300)
        }
    
    def get_logging_settings(self) -> Dict[str, Any]:
        """Get logging configuration from config"""
        return {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file_rotation": True,
            "max_file_size": "10MB",
            "backup_count": 5
        }
    
    def get_output_settings(self) -> Dict[str, Any]:
        """Get output organization settings from config"""
        return {
            "training_cycles": {
                "base_dir": "output/training_cycles",
                "timestamp_format": "%Y%m%d_%H%M%S",
                "subdirs": ["logs", "plots", "results", "reports"]
            },
            "inference_results": {
                "base_dir": "output/inference_results"
            }
        }
    
    def update_config_value(self, key: str, value: Any):
        """Update a configuration value and reinitialize training config"""
        self.config_data[key] = value
        
        # Reinitialize training configuration
        self._init_training_config()
    
    def save_config_to_yaml(self, output_path: str = None):
        """Save current configuration to YAML file"""
        if output_path is None:
            output_path = self.project_root / "models" / "yolov8_kidney_stone_v3" / "configs" / "model_config_v3.yaml"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config_data, file, default_flow_style=False, sort_keys=False)
            print(f"✅ Configuration saved to {output_path}")
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")
    
    # Legacy methods for backward compatibility
    def get_training_output_dir(self) -> Path:
        """Get the base directory for training outputs (legacy)"""
        return self.output_dir / "training_cycles"
    
    def get_test_output_dir(self) -> Path:
        """Get the base directory for test/evaluation results (legacy)"""
        return self.output_dir / "training_cycles"
    
    def get_plots_output_dir(self) -> Path:
        """Get the base directory for generated plots (legacy)"""
        return self.output_dir / "training_cycles"
    
    def get_reports_output_dir(self) -> Path:
        """Get the base directory for analysis reports (legacy)"""
        return self.output_dir / "training_cycles"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "project_root": str(self.project_root),
            "data_dir": str(self.data_dir),
            "models_dir": str(self.models_dir),
            "dataset_path": str(self.dataset_path),
            "training_config": self.training_config,
            "gui_config": self.gui_config,
            "yaml_config": self.config_data
        }

# Global configuration instance
config = Config()
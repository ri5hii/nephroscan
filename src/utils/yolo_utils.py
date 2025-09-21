import os
import cv2
import numpy as np
import torch
from pathlib import Path
import shutil
from datetime import datetime
import yaml
import json
from ultralytics import YOLO
from PIL import Image
import math


class YOLOTrainer:
    """Class for training YOLO models on kidney stone datasets"""
    
    def __init__(self, config=None):
        """Initialize YOLO trainer with configuration.
        
        Args:
            config: Configuration object with paths and settings
        """
        self.model = None
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Use config's models directory if provided, otherwise fallback
        if config:
            self.models_dir = config.models_dir
            self.pretrained_path = config.yolo_pretrained
        else:
            self.models_dir = Path("models")
            self.models_dir.mkdir(exist_ok=True)
            self.pretrained_path = self.models_dir / "pretrained" / "yolov8n.pt"
        
    def prepare_dataset(self, dataset_path, progress_callback=None):
        """Prepare dataset in YOLO format"""
        if progress_callback:
            progress_callback("Preparing dataset structure...")
        
        dataset_path = Path(dataset_path)
        
        # Expected structure: dataset/train/images, dataset/train/labels, dataset/val/images, dataset/val/labels
        if not (dataset_path / "train" / "images").exists():
            # If not in YOLO format, create a basic structure
            if progress_callback:
                progress_callback("Converting dataset to YOLO format...")
            
            # Create basic structure (assuming images are in root folder)
            train_dir = dataset_path / "train"
            val_dir = dataset_path / "val"
            
            for split_dir in [train_dir, val_dir]:
                (split_dir / "images").mkdir(parents=True, exist_ok=True)
                (split_dir / "labels").mkdir(parents=True, exist_ok=True)
            
            # Move images (basic split - 80% train, 20% val)
            image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
            if image_files:
                split_idx = int(len(image_files) * 0.8)
                
                for i, img_file in enumerate(image_files):
                    if i < split_idx:
                        dest_dir = train_dir / "images"
                    else:
                        dest_dir = val_dir / "images"
                    
                    shutil.copy2(img_file, dest_dir / img_file.name)
                    
                    # Create empty label file if not exists
                    label_file = img_file.with_suffix('.txt')
                    if label_file.exists():
                        if i < split_idx:
                            dest_label_dir = train_dir / "labels"
                        else:
                            dest_label_dir = val_dir / "labels"
                        shutil.copy2(label_file, dest_label_dir / label_file.name)
        
        # Create dataset.yaml
        dataset_yaml = dataset_path / "dataset.yaml"
        yaml_content = {
            'path': str(dataset_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'names': {0: 'kidney_stone'},
            'nc': 1
        }
        
        with open(dataset_yaml, 'w') as f:
            yaml.dump(yaml_content, f)
        
        if progress_callback:
            progress_callback("Dataset preparation completed")
        
        return str(dataset_yaml)
    
    def train(self, dataset_path, epochs=50, img_size=640, batch_size=16, progress_callback=None):
        """Train YOLO model on kidney stone dataset"""
        try:
            if progress_callback:
                progress_callback("Initializing YOLO model...")
            
            # Prepare dataset
            dataset_yaml = self.prepare_dataset(dataset_path, progress_callback)
            
            # Initialize model
            self.model = YOLO(str(self.pretrained_path))  # Use organized pretrained model
            
            if progress_callback:
                progress_callback(f"Starting training for {epochs} epochs...")
            
            # Training parameters
            train_results = self.model.train(
                data=dataset_yaml,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                save=True,
                project=str(self.results_dir),
                name='kidney_stone_training',
                exist_ok=True,
                verbose=True
            )
            
            # Save model
            model_save_path = self.results_dir / "kidney_stone_model.pt"
            self.model.save(str(model_save_path))
            
            if progress_callback:
                progress_callback("Training completed successfully!")
            
            # Prepare results
            results = {
                'status': 'completed',
                'model_path': str(model_save_path),
                'epochs': epochs,
                'dataset_path': dataset_path,
                'timestamp': datetime.now().isoformat(),
                'metrics': self.extract_training_metrics(train_results)
            }
            
            return results
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Training failed: {str(e)}")
            raise e
    
    def extract_training_metrics(self, train_results):
        """Extract metrics from training results"""
        try:
            if hasattr(train_results, 'results_dict'):
                metrics = train_results.results_dict
            else:
                metrics = {}
            
            return {
                'final_loss': metrics.get('train/box_loss', 'N/A'),
                'final_precision': metrics.get('metrics/precision(B)', 'N/A'),
                'final_recall': metrics.get('metrics/recall(B)', 'N/A'),
                'final_map50': metrics.get('metrics/mAP50(B)', 'N/A'),
                'final_map50_95': metrics.get('metrics/mAP50-95(B)', 'N/A')
            }
        except:
            return {'note': 'Metrics extraction failed, but training completed'}


class YOLOTester:
    """Class for testing YOLO models on kidney stone datasets"""
    
    def __init__(self):
        """Initialize YOLO tester for model evaluation."""
        self.model = None
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
    
    def test(self, dataset_path, model_path=None, progress_callback=None):
        """Test YOLO model on dataset"""
        try:
            if progress_callback:
                progress_callback("Loading model for testing...")
            
            # Load model
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                # Use default trained model or pretrained
                default_model = self.results_dir / "kidney_stone_model.pt"
                if default_model.exists():
                    self.model = YOLO(str(default_model))
                else:
                    self.model = YOLO(str(self.pretrained_path))
            
            if progress_callback:
                progress_callback("Preparing test dataset...")
            
            # Prepare dataset for validation
            dataset_path = Path(dataset_path)
            val_images_path = dataset_path / "val" / "images"
            
            if not val_images_path.exists():
                # If no validation set, use test images from root
                test_images = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
            else:
                test_images = list(val_images_path.glob("*.jpg")) + list(val_images_path.glob("*.png"))
            
            if not test_images:
                raise ValueError("No test images found in dataset")
            
            if progress_callback:
                progress_callback(f"Testing on {len(test_images)} images...")
            
            # Run validation
            val_results = self.model.val(
                data=str(dataset_path / "dataset.yaml") if (dataset_path / "dataset.yaml").exists() else None,
                save=True,
                project=str(self.results_dir),
                name='kidney_stone_testing',
                exist_ok=True
            )
            
            if progress_callback:
                progress_callback("Extracting test metrics...")
            
            # Extract metrics
            metrics = self.extract_test_metrics(val_results)
            
            if progress_callback:
                progress_callback("Testing completed successfully!")
            
            results = {
                'status': 'completed',
                'model_path': model_path,
                'dataset_path': str(dataset_path),
                'num_test_images': len(test_images),
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            }
            
            return results
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Testing failed: {str(e)}")
            raise e
    
    def extract_test_metrics(self, val_results):
        """Extract metrics from validation results"""
        try:
            if hasattr(val_results, 'results_dict'):
                metrics_dict = val_results.results_dict
            else:
                metrics_dict = {}
            
            return {
                'precision': metrics_dict.get('metrics/precision(B)', 'N/A'),
                'recall': metrics_dict.get('metrics/recall(B)', 'N/A'),
                'map50': metrics_dict.get('metrics/mAP50(B)', 'N/A'),
                'map50_95': metrics_dict.get('metrics/mAP50-95(B)', 'N/A'),
                'fitness': metrics_dict.get('fitness', 'N/A')
            }
        except:
            return {'note': 'Metrics extraction failed, but testing completed'}


class YOLOInference:
    """Class for running inference on single images"""
    
    def __init__(self):
        """Initialize YOLO inference engine."""
        self.model = None
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
    
    def predict(self, image_path, model_path=None, confidence_threshold=0.25, progress_callback=None):
        """Run inference on a single image"""
        try:
            if progress_callback:
                progress_callback("Loading model for inference...")
            
            # Load model
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                # Use default trained model or pretrained
                default_model = self.results_dir / "kidney_stone_model.pt"
                if default_model.exists():
                    self.model = YOLO(str(default_model))
                else:
                    self.model = YOLO(str(self.pretrained_path))
            
            if progress_callback:
                progress_callback("Running inference on image...")
            
            # Run inference
            results = self.model(image_path, conf=confidence_threshold, save=True, 
                               project=str(self.results_dir), name='inference', exist_ok=True)
            
            if progress_callback:
                progress_callback("Processing detection results...")
            
            # Process results
            detections = self.process_detections(results[0], image_path)
            
            # Save annotated image
            output_image_path = self.save_annotated_image(results[0], image_path)
            
            if progress_callback:
                progress_callback("Inference completed successfully!")
            
            result_data = {
                'status': 'completed',
                'original_image_path': str(image_path),
                'output_image_path': output_image_path,
                'model_path': model_path or 'default',
                'confidence_threshold': confidence_threshold,
                'timestamp': datetime.now().isoformat(),
                'detections': detections,
                'summary': {
                    'stones_detected': len(detections) > 0,
                    'num_stones': len(detections),
                    'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0
                }
            }
            
            return result_data
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Inference failed: {str(e)}")
            raise e
    
    def process_detections(self, result, image_path):
        """Process YOLO detection results"""
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes
            
            for i in range(len(boxes.xyxy)):
                # Extract bounding box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                
                # Calculate center and size
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                # Calculate stone size (area in pixels)
                area = width * height
                
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [float(center_x), float(center_y)],
                    'size': float(area),
                    'width': float(width),
                    'height': float(height),
                    'confidence': confidence,
                    'class': 'kidney_stone'
                }
                
                detections.append(detection)
        
        return detections
    
    def save_annotated_image(self, result, original_image_path):
        """Save image with detection annotations"""
        # Get the annotated image from results
        annotated_img = result.plot()
        
        # Convert from BGR to RGB (OpenCV to PIL)
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Save annotated image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"kidney_stone_detection_{timestamp}.jpg"
        output_path = self.results_dir / output_filename
        
        pil_img = Image.fromarray(annotated_img_rgb)
        pil_img.save(output_path, "JPEG", quality=95)
        
        return str(output_path)
    
    def calculate_stone_size_mm(self, pixel_size, calibration_factor=1.0):
        """Convert pixel size to mm (requires calibration factor)"""
        # This is a placeholder - in real applications, you'd need proper calibration
        # based on imaging parameters, patient distance, etc.
        return pixel_size * calibration_factor


# Utility functions
def create_sample_dataset(output_dir, num_images=10):
    """Create a sample dataset structure for testing"""
    output_dir = Path(output_dir)
    
    # Create directory structure
    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Create sample images and labels
    for split in ['train', 'val']:
        split_count = num_images if split == 'train' else max(1, num_images // 4)
        
        for i in range(split_count):
            # Create a dummy image
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img_path = output_dir / split / 'images' / f'sample_{i:03d}.jpg'
            cv2.imwrite(str(img_path), img)
            
            # Create a dummy label (random bounding box)
            label_path = output_dir / split / 'labels' / f'sample_{i:03d}.txt'
            
            # Random bounding box in YOLO format (class x_center y_center width height)
            x_center = np.random.uniform(0.2, 0.8)
            y_center = np.random.uniform(0.2, 0.8)
            width = np.random.uniform(0.1, 0.3)
            height = np.random.uniform(0.1, 0.3)
            
            with open(label_path, 'w') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Create dataset.yaml
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'kidney_stone'},
        'nc': 1
    }
    
    with open(output_dir / 'dataset.yaml', 'w') as f:
        yaml.dump(yaml_content, f)
    
    print(f"Sample dataset created at: {output_dir}")
    return str(output_dir)
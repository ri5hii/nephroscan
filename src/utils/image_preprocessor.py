#!/usr/bin/env python3
"""
Unified Image Preprocessing Module for NephroScan v3

This module provides intelligent preprocessing for medical images to ensure
they match the training data format for optimal kidney stone detection.

Key Features:
- Automatic dimension standardization (391x320 pixels)
- Medical image enhancement (CLAHE, bilateral filtering)
- Color scheme compatibility checking and validation
- Dynamic path handling (no hardcoded paths)
- Conditional processing (only when needed)
- Coordinate scaling for detection mapping
- Comprehensive preprocessing validation

Author: NephroScan Development Team
Version: 3.0.0
"""

import cv2
import numpy as np
import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import tempfile
import time

class ImagePreprocessor:
    """
    Unified image preprocessing system for medical image standardization.
    
    This class handles all aspects of image preprocessing including:
    - Automatic preprocessing detection
    - Medical image enhancement
    - Color scheme validation
    - Dynamic path management
    - Coordinate scaling for detection mapping
    """
    
    # Target specifications based on training data analysis
    TARGET_WIDTH = 391
    TARGET_HEIGHT = 320
    TARGET_CHANNELS = 3
    TARGET_BRIGHTNESS = 82.37  # From training data analysis
    TARGET_CONTRAST = 62.68    # From training data analysis
    JPEG_QUALITY = 95
    
    # Tolerance thresholds for validation
    BRIGHTNESS_TOLERANCE = 10.0
    CONTRAST_TOLERANCE = 15.0
    
    def __init__(self, logger: Optional[logging.Logger] = None, 
                 output_dir: Optional[str] = None):
        """
        Initialize the unified image preprocessor
        
        Args:
            logger: Optional logger instance for logging operations
            output_dir: Base output directory (defaults to output/preprocessing)
        """
        self.logger = logger or logging.getLogger(__name__)
        self.target_dimensions = (self.TARGET_WIDTH, self.TARGET_HEIGHT)
        
        # Set up dynamic output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Use a default temporary directory that can be cleaned up
            self.output_dir = Path("output") / "preprocessing"
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"ImagePreprocessor initialized with output dir: {self.output_dir}")
    
    def detect_image_type(self, image: np.ndarray) -> str:
        """
        Detect the type of medical image based on characteristics.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Image type: 'CT_SCAN', 'LOW_QUALITY', 'ULTRASOUND', or 'STANDARD_MEDICAL'
        """
        height, width = image.shape[:2]
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate image characteristics
        contrast = gray.std()
        brightness_mean = gray.mean()
        dark_ratio = np.sum(gray < 50) / gray.size
        bright_ratio = np.sum(gray > 200) / gray.size
        resolution = width * height
        
        self.logger.debug(f"Image analysis - Resolution: {width}x{height}, Contrast: {contrast:.2f}, "
                         f"Brightness: {brightness_mean:.2f}, Dark ratio: {dark_ratio:.3f}, "
                         f"Bright ratio: {bright_ratio:.3f}")
        
        # CT scan characteristics
        if (contrast > 40 and dark_ratio > 0.2 and bright_ratio > 0.02 and 
            min(width, height) > 400 and brightness_mean < 120):
            return "CT_SCAN"
        
        # Low quality/screenshot characteristics
        elif (contrast < 35 or resolution < 160000 or 
              (bright_ratio < 0.01 and contrast < 50)):
            return "LOW_QUALITY"
        
        # Ultrasound characteristics (lower contrast, specific intensity pattern)
        elif (contrast < 45 and dark_ratio > 0.4 and brightness_mean < 80):
            return "ULTRASOUND"
        
        # Standard medical image
        else:
            return "STANDARD_MEDICAL"
    
    def assess_image_quality(self, image: np.ndarray) -> str:
        """
        Assess the quality level of the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Quality level: 'HIGH', 'MEDIUM', or 'LOW'
        """
        height, width = image.shape[:2]
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate quality metrics
        resolution = width * height
        contrast = gray.std()
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # High quality: high resolution, good contrast, sharp
        if resolution > 300000 and contrast > 50 and sharpness > 100:
            return "HIGH"
        
        # Low quality: low resolution, poor contrast, blurry
        elif resolution < 100000 or contrast < 25 or sharpness < 50:
            return "LOW"
        
        # Medium quality
        else:
            return "MEDIUM"

    def analyze_training_data_characteristics(self, training_dir: str = "data/train/images") -> Dict[str, float]:
        """
        Analyze multiple training images to determine target characteristics.
        This provides the baseline for color scheme matching.
        
        Args:
            training_dir: Path to training images directory
            
        Returns:
            Dictionary with target characteristics
        """
        if not os.path.exists(training_dir):
            self.logger.warning(f"Training directory not found: {training_dir}")
            return {
                'brightness': self.TARGET_BRIGHTNESS,
                'contrast': self.TARGET_CONTRAST,
                'red_mean': self.TARGET_BRIGHTNESS,
                'green_mean': self.TARGET_BRIGHTNESS,
                'blue_mean': self.TARGET_BRIGHTNESS
            }
        
        training_files = [f for f in os.listdir(training_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:10]
        
        if not training_files:
            self.logger.warning("No training images found for analysis")
            return {
                'brightness': self.TARGET_BRIGHTNESS,
                'contrast': self.TARGET_CONTRAST,
                'red_mean': self.TARGET_BRIGHTNESS,
                'green_mean': self.TARGET_BRIGHTNESS,
                'blue_mean': self.TARGET_BRIGHTNESS
            }
        
        brightness_values = []
        contrast_values = []
        red_means = []
        green_means = []
        blue_means = []
        
        for filename in training_files:
            filepath = os.path.join(training_dir, filename)
            img = cv2.imread(filepath)
            if img is None:
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            brightness_values.append(np.mean(gray))
            contrast_values.append(np.std(gray))
            red_means.append(np.mean(img_rgb[:, :, 0]))
            green_means.append(np.mean(img_rgb[:, :, 1]))
            blue_means.append(np.mean(img_rgb[:, :, 2]))
        
        if brightness_values:
            result = {
                'brightness': np.mean(brightness_values),
                'contrast': np.mean(contrast_values),
                'red_mean': np.mean(red_means),
                'green_mean': np.mean(green_means),
                'blue_mean': np.mean(blue_means)
            }
            self.logger.info(f"Training data analysis: brightness={result['brightness']:.2f}, contrast={result['contrast']:.2f}")
            return result
        else:
            return {
                'brightness': self.TARGET_BRIGHTNESS,
                'contrast': self.TARGET_CONTRAST,
                'red_mean': self.TARGET_BRIGHTNESS,
                'green_mean': self.TARGET_BRIGHTNESS,
                'blue_mean': self.TARGET_BRIGHTNESS
            }
    
    def get_dynamic_output_path(self, input_path: str, suffix: str = "_preprocessed") -> str:
        """
        Generate dynamic output path based on input image path.
        
        Args:
            input_path: Path to input image
            suffix: Suffix to add to filename
            
        Returns:
            Full path for preprocessed image
        """
        input_path = Path(input_path)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create filename with timestamp to avoid conflicts
        output_filename = f"{input_path.stem}{suffix}_{timestamp}.jpg"
        output_path = self.output_dir / output_filename
        
        return str(output_path)
    
    def needs_preprocessing_comprehensive(self, image_path: str) -> Dict[str, Any]:
        """
        Comprehensive check if an image needs preprocessing.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with preprocessing analysis results
        """
        try:
            # Load image to check properties
            image = cv2.imread(image_path)
            if image is None:
                self.logger.warning(f"Could not load image: {image_path}")
                return {
                    'needs_preprocessing': True,
                    'reason': 'Could not load image',
                    'dimensions_match': False,
                    'color_compatible': False,
                    'current_dimensions': None,
                    'target_dimensions': self.target_dimensions
                }
            
            height, width, channels = image.shape
            current_dimensions = (width, height)
            
            # Check dimensions
            dimensions_match = current_dimensions == self.target_dimensions
            channels_match = channels == self.TARGET_CHANNELS
            
            # Check color compatibility
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            current_brightness = np.mean(gray)
            current_contrast = np.std(gray)
            
            brightness_compatible = abs(current_brightness - self.TARGET_BRIGHTNESS) <= self.BRIGHTNESS_TOLERANCE
            contrast_compatible = abs(current_contrast - self.TARGET_CONTRAST) <= self.CONTRAST_TOLERANCE
            color_compatible = brightness_compatible and contrast_compatible
            
            needs_preprocessing = not (dimensions_match and channels_match and color_compatible)
            
            result = {
                'needs_preprocessing': needs_preprocessing,
                'reason': self._get_preprocessing_reason(dimensions_match, channels_match, color_compatible),
                'dimensions_match': dimensions_match,
                'channels_match': channels_match,
                'color_compatible': color_compatible,
                'current_dimensions': current_dimensions,
                'target_dimensions': self.target_dimensions,
                'current_brightness': current_brightness,
                'target_brightness': self.TARGET_BRIGHTNESS,
                'current_contrast': current_contrast,
                'target_contrast': self.TARGET_CONTRAST,
                'brightness_compatible': brightness_compatible,
                'contrast_compatible': contrast_compatible
            }
            
            if not needs_preprocessing:
                self.logger.info(f"Image already optimized: {width}x{height}, brightness={current_brightness:.1f}, contrast={current_contrast:.1f}")
            else:
                self.logger.info(f"Preprocessing needed: {result['reason']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing image {image_path}: {str(e)}")
            return {
                'needs_preprocessing': True,
                'reason': f'Analysis error: {str(e)}',
                'dimensions_match': False,
                'color_compatible': False,
                'current_dimensions': None,
                'target_dimensions': self.target_dimensions
            }
    
    def _get_preprocessing_reason(self, dimensions_match: bool, channels_match: bool, color_compatible: bool) -> str:
        """Get human-readable reason for preprocessing"""
        reasons = []
        if not dimensions_match:
            reasons.append("dimensions need adjustment")
        if not channels_match:
            reasons.append("channel count mismatch")
        if not color_compatible:
            reasons.append("color characteristics need enhancement")
        
        if not reasons:
            return "no preprocessing needed"
        
        return "; ".join(reasons)
    
    def needs_preprocessing(self, image_path: str) -> bool:
        """
        Simple boolean check if an image needs preprocessing (backward compatibility)
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if preprocessing is needed, False otherwise
        """
        analysis = self.needs_preprocessing_comprehensive(image_path)
        return analysis.get('needs_preprocessing', True)
    
    def preprocess_ct_scan(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Special preprocessing for CT scans to preserve kidney stone visibility.
        Now includes final resize to training dimensions while preserving enhanced features.
        
        Args:
            image: Input CT scan image
            
        Returns:
            Tuple of (processed_image, processing_info)
        """
        original_height, original_width = image.shape[:2]
        
        # 1. First preserve higher resolution for enhancement (minimum 640px on smaller side)
        min_dimension = min(original_width, original_height)
        if min_dimension > 640:
            # Calculate scale to maintain aspect ratio with 640px minimum
            scale = 640 / min_dimension
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # Use high-quality downsampling for intermediate processing
            intermediate = cv2.resize(image, (new_width, new_height), 
                               interpolation=cv2.INTER_LANCZOS4)
        else:
            intermediate = image.copy()
            new_width, new_height = original_width, original_height
        
        # 2. Convert to grayscale if needed for processing
        if len(intermediate.shape) == 3:
            gray = cv2.cvtColor(intermediate, cv2.COLOR_BGR2GRAY)
        else:
            gray = intermediate.copy()
        
        # 3. Gentle sharpening to enhance small bright features (kidney stones)
        # Use unsharp masking for medical images
        gaussian = cv2.GaussianBlur(gray, (3, 3), 1.0)
        sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        
        # 4. Very mild contrast enhancement (preserve natural CT contrast)
        # Use gentle linear scaling instead of aggressive CLAHE
        enhanced = cv2.convertScaleAbs(sharpened, alpha=1.1, beta=5)
        
        # 5. Convert back to BGR for consistency
        if len(image.shape) == 3:
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            enhanced_bgr = enhanced
        
        # 6. CRITICAL: Final resize to training dimensions while preserving enhanced features
        # Use LANCZOS for high-quality downsampling that preserves small features
        final_image = cv2.resize(enhanced_bgr, (self.TARGET_WIDTH, self.TARGET_HEIGHT), 
                               interpolation=cv2.INTER_LANCZOS4)
        
        processing_info = {
            'method': 'CT_preservation_hybrid',
            'original_size': (original_width, original_height),
            'intermediate_size': (new_width, new_height),
            'processed_size': (self.TARGET_WIDTH, self.TARGET_HEIGHT),
            'scale_factor': (self.TARGET_WIDTH / original_width, self.TARGET_HEIGHT / original_height),
            'enhancement': 'gentle_sharpening_then_resize',
            'contrast_adjustment': 'mild_linear_with_model_compatibility'
        }
        
        self.logger.info(f"CT hybrid preprocessing: {original_width}x{original_height} -> {new_width}x{new_height} -> {self.TARGET_WIDTH}x{self.TARGET_HEIGHT}")
        return final_image, processing_info
    
    def preprocess_standard_medical(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Standard preprocessing for regular medical images (current approach).
        
        Args:
            image: Input medical image
            
        Returns:
            Tuple of (processed_image, processing_info)
        """
        original_height, original_width = image.shape[:2]
        
        # Use our current proven approach
        # 1. Resize to training data dimensions
        resized = cv2.resize(image, (self.TARGET_WIDTH, self.TARGET_HEIGHT), 
                           interpolation=cv2.INTER_LANCZOS4)
        
        # 2. Apply medical enhancement
        enhanced = self._enhance_medical_image(resized)
        
        processing_info = {
            'method': 'standard_medical',
            'original_size': (original_width, original_height),
            'processed_size': (self.TARGET_WIDTH, self.TARGET_HEIGHT),
            'scale_factor': (self.TARGET_WIDTH / original_width, self.TARGET_HEIGHT / original_height),
            'enhancement': 'CLAHE_bilateral',
            'target_match': True
        }
        
        return enhanced, processing_info
    
    def preprocess_low_quality(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Enhanced preprocessing for low-quality images.
        
        Args:
            image: Input low-quality image
            
        Returns:
            Tuple of (processed_image, processing_info)
        """
        original_height, original_width = image.shape[:2]
        
        # 1. Resize to training dimensions
        resized = cv2.resize(image, (self.TARGET_WIDTH, self.TARGET_HEIGHT), 
                           interpolation=cv2.INTER_LANCZOS4)
        
        # 2. Aggressive enhancement for low-quality images
        enhanced = self._enhance_medical_image(resized)
        
        # 3. Additional noise reduction for low-quality images
        if len(enhanced.shape) == 3:
            # Apply additional bilateral filtering
            denoised = cv2.bilateralFilter(enhanced, 9, 80, 80)
        else:
            denoised = enhanced
        
        processing_info = {
            'method': 'low_quality_enhancement',
            'original_size': (original_width, original_height),
            'processed_size': (self.TARGET_WIDTH, self.TARGET_HEIGHT),
            'scale_factor': (self.TARGET_WIDTH / original_width, self.TARGET_HEIGHT / original_height),
            'enhancement': 'aggressive_CLAHE_bilateral_denoise'
        }
        
        return denoised, processing_info

    def preprocess_image(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Preprocess an image to match the target format
        
        Args:
            input_path: Path to the input image
            output_path: Optional path for the output image. If None, overwrites input
            
        Returns:
            Path to the preprocessed image
            
        Raises:
            ValueError: If image cannot be loaded or processed
        """
        try:
            # Load the image
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError(f"Could not load image from: {input_path}")
            
            original_height, original_width, original_channels = image.shape
            self.logger.info(f"Original image: {original_width}x{original_height}x{original_channels}")
            
            # 1. Detect image type and quality
            image_type = self.detect_image_type(image)
            image_quality = self.assess_image_quality(image)
            
            self.logger.info(f"Detected image type: {image_type}, quality: {image_quality}")
            
            # 2. Convert to standard BGR format first
            if original_channels == 1:
                # Convert grayscale to 3-channel
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                self.logger.info("Converted grayscale to 3-channel image")
            elif original_channels == 4:
                # Convert RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                self.logger.info("Converted RGBA to RGB")
            
            # 3. Apply adaptive preprocessing based on image type and quality
            if image_type == "CT_SCAN" and image_quality == "HIGH":
                processed_image, processing_info = self.preprocess_ct_scan(image)
                self.logger.info("Applied CT scan preservation preprocessing")
            elif image_type == "LOW_QUALITY" or image_quality == "LOW":
                processed_image, processing_info = self.preprocess_low_quality(image)
                self.logger.info("Applied low-quality enhancement preprocessing")
            else:
                # Standard medical image processing (our current proven approach)
                processed_image, processing_info = self.preprocess_standard_medical(image)
                self.logger.info("Applied standard medical preprocessing")
            
            # 4. Determine output path
            if output_path is None:
                output_path = input_path
            
            # 5. Save with high quality
            save_params = [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY]
            success = cv2.imwrite(output_path, processed_image, save_params)
            
            if not success:
                raise ValueError(f"Failed to save preprocessed image to: {output_path}")
            
            # 6. Log processing details
            final_height, final_width = processed_image.shape[:2]
            self.logger.info(f"Preprocessing complete: {processing_info['method']}")
            self.logger.info(f"Final image: {final_width}x{final_height}")
            self.logger.info(f"Successfully preprocessed image: {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")
    
    def get_coordinate_scaling(self, input_path: str, processing_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Get coordinate scaling factors for mapping detections back to original image.
        
        Args:
            input_path: Path to original image
            processing_info: Information from preprocessing
            
        Returns:
            Dictionary with scaling factors
        """
        try:
            # Load original image to get dimensions
            original_image = cv2.imread(input_path)
            if original_image is None:
                raise ValueError(f"Could not load original image: {input_path}")
            
            original_height, original_width = original_image.shape[:2]
            processed_width, processed_height = processing_info['processed_size']
            
            width_scale = original_width / processed_width
            height_scale = original_height / processed_height
            
            return {
                'width_scale': width_scale,
                'height_scale': height_scale,
                'original_size': (original_width, original_height),
                'processed_size': (processed_width, processed_height),
                'method': processing_info.get('method', 'unknown')
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating coordinate scaling: {str(e)}")
            # Return default scaling (assumes standard preprocessing)
            return {
                'width_scale': 1.0,
                'height_scale': 1.0,
                'original_size': (391, 320),
                'processed_size': (391, 320),
                'method': 'default'
            }

    def _enhance_medical_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply medical image enhancement techniques
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Enhanced image
        """
        try:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # This enhances contrast while preventing over-amplification
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            
            # Convert back to 3-channel
            enhanced_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            
            # Slight denoising while preserving edges
            denoised = cv2.bilateralFilter(enhanced_image, 5, 50, 50)
            
            return denoised
            
        except Exception as e:
            self.logger.warning(f"Image enhancement failed, using original: {str(e)}")
            return image
    
    def create_preprocessed_copy(self, input_path: str, output_dir: str) -> str:
        """
        Create a preprocessed copy of an image in a specified directory
        
        Args:
            input_path: Path to the input image
            output_dir: Directory to save the preprocessed copy
            
        Returns:
            Path to the preprocessed copy
        """
        try:
            input_path = Path(input_path)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename with preprocessing suffix
            stem = input_path.stem
            suffix = input_path.suffix
            output_filename = f"{stem}_preprocessed{suffix}"
            output_path = output_dir / output_filename
            
            return self.preprocess_image(str(input_path), str(output_path))
            
        except Exception as e:
            self.logger.error(f"Error creating preprocessed copy: {str(e)}")
            raise
    
    def get_image_info(self, image_path: str) -> dict:
        """
        Get detailed information about an image
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with image information
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image"}
            
            height, width, channels = image.shape
            file_size = Path(image_path).stat().st_size
            
            return {
                "width": width,
                "height": height,
                "channels": channels,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "matches_target": (width, height) == self.target_dimensions and channels == self.TARGET_CHANNELS,
                "needs_preprocessing": self.needs_preprocessing(image_path)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def process_for_inference(self, input_path: str, force_preprocessing: bool = False) -> Dict[str, Any]:
        """
        Unified preprocessing workflow for inference.
        
        This is the main method that should be used for all single inference preprocessing.
        It automatically determines if preprocessing is needed and handles all aspects
        including path management, validation, and coordinate scaling preparation.
        
        Args:
            input_path: Path to the input image
            force_preprocessing: Force preprocessing even if not needed
            
        Returns:
            Dictionary containing:
            - final_image_path: Path to the final image (preprocessed or original)
            - was_preprocessed: Boolean indicating if preprocessing was applied
            - preprocessing_info: Analysis results
            - scale_factors: Coordinate scaling factors (width_scale, height_scale)
            - original_dimensions: Original image dimensions
            - final_dimensions: Final image dimensions
        """
        try:
            self.logger.info(f"Starting unified preprocessing workflow for: {input_path}")
            
            # Comprehensive preprocessing analysis
            analysis = self.needs_preprocessing_comprehensive(input_path)
            
            # Determine if we should preprocess
            should_preprocess = force_preprocessing or analysis['needs_preprocessing']
            
            if not should_preprocess:
                # No preprocessing needed
                self.logger.info("No preprocessing required - using original image")
                
                # Get original dimensions for scaling factors
                image = cv2.imread(input_path)
                if image is not None:
                    height, width = image.shape[:2]
                    original_dims = (width, height)
                    final_dims = original_dims
                else:
                    original_dims = analysis.get('current_dimensions', (0, 0))
                    final_dims = original_dims
                
                return {
                    'final_image_path': input_path,
                    'was_preprocessed': False,
                    'preprocessing_info': analysis,
                    'scale_factors': (1.0, 1.0),  # No scaling needed
                    'original_dimensions': original_dims,
                    'final_dimensions': final_dims,
                    'success': True,
                    'message': 'Image already optimized for inference'
                }
            
            # Preprocessing is needed
            self.logger.info(f"Preprocessing required: {analysis['reason']}")
            
            # Generate dynamic output path
            output_path = self.get_dynamic_output_path(input_path)
            
            # Get original dimensions before preprocessing
            image = cv2.imread(input_path)
            if image is None:
                return {
                    'final_image_path': input_path,
                    'was_preprocessed': False,
                    'preprocessing_info': analysis,
                    'scale_factors': (1.0, 1.0),
                    'original_dimensions': (0, 0),
                    'final_dimensions': (0, 0),
                    'success': False,
                    'message': 'Could not load input image'
                }
            
            original_height, original_width = image.shape[:2]
            original_dims = (original_width, original_height)
            
            # NEW: Use adaptive preprocessing instead of old hardcoded approach
            image_type = self.detect_image_type(image)
            image_quality = self.assess_image_quality(image)
            
            self.logger.info(f"Detected image type: {image_type}, quality: {image_quality}")
            
            # Apply appropriate preprocessing based on detection
            if image_type == "CT_SCAN" and image_quality == "HIGH":
                processed_image, processing_info = self.preprocess_ct_scan(image)
                self.logger.info("Applied CT scan preservation preprocessing")
            elif image_type == "LOW_QUALITY" or image_quality == "LOW":
                processed_image, processing_info = self.preprocess_low_quality(image)
                self.logger.info("Applied low-quality enhancement preprocessing")
            else:
                processed_image, processing_info = self.preprocess_standard_medical(image)
                self.logger.info("Applied standard medical preprocessing")
            
            # Save the processed image
            save_params = [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY]
            success = cv2.imwrite(output_path, processed_image, save_params)
            
            if not success:
                raise ValueError(f"Failed to save processed image to: {output_path}")
            
            # Calculate scaling factors using the actual processed dimensions
            final_height, final_width = processed_image.shape[:2]
            final_dims = (final_width, final_height)
            width_scale = original_width / final_width
            height_scale = original_height / final_height
            
            self.logger.info(f"Preprocessing completed successfully")
            self.logger.info(f"Scale factors: width={width_scale:.3f}, height={height_scale:.3f}")
            
            return {
                'final_image_path': output_path,
                'was_preprocessed': True,
                'preprocessing_info': processing_info,
                'scale_factors': (width_scale, height_scale),
                'original_dimensions': original_dims,
                'final_dimensions': final_dims,
                'success': True,
                'message': f'Image preprocessed successfully with {processing_info["method"]}: {analysis["reason"]}'
            }
            
        except Exception as e:
            error_msg = f"Preprocessing workflow failed: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                'final_image_path': input_path,
                'was_preprocessed': False,
                'preprocessing_info': {},
                'scale_factors': (1.0, 1.0),
                'original_dimensions': (0, 0),
                'final_dimensions': (0, 0),
                'success': False,
                'message': error_msg
            }


def preprocess_for_inference(image_path: str, logger: Optional[logging.Logger] = None) -> str:
    """
    Convenience function to preprocess an image for inference
    
    Args:
        image_path: Path to the image to preprocess
        logger: Optional logger instance
        
    Returns:
        Path to the preprocessed image (may be the same as input if no preprocessing needed)
    """
    preprocessor = ImagePreprocessor(logger)
    
    if not preprocessor.needs_preprocessing(image_path):
        return image_path
    
    # Create preprocessed copy in the same directory
    input_path = Path(image_path)
    output_path = input_path.parent / f"{input_path.stem}_preprocessed{input_path.suffix}"
    
    return preprocessor.preprocess_image(image_path, str(output_path))
import os
from typing import List, Optional, Union, Dict, Any
from PIL import Image
import numpy as np
import torch
from pathlib import Path
import mimetypes
import logging

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class Validators:
    """Utility class for input validation"""

    @staticmethod
    def validate_image_file(file_path: Union[str, Path]) -> str:
        """
        Validate image file existence and format
        Args:
            file_path: Path to image file
        Returns:
            Validated file path as string
        Raises:
            ValidationError: If validation fails
        """
        path = Path(file_path)
        
        # Check existence
        if not path.exists():
            raise ValidationError(f"Image file not found: {file_path}")
            
        # Check if it's a file
        if not path.is_file():
            raise ValidationError(f"Not a file: {file_path}")
            
        # Check file type
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type or not mime_type.startswith('image/'):
            raise ValidationError(f"Not an image file: {file_path}")
            
        # Try opening the image
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            raise ValidationError(f"Invalid image file: {str(e)}")
            
        return str(path)

    @staticmethod
    def validate_mask(mask: np.ndarray, image_shape: tuple) -> np.ndarray:
        """
        Validate segmentation mask
        Args:
            mask: Input mask array
            image_shape: Expected shape (H, W)
        Returns:
            Validated mask array
        Raises:
            ValidationError: If validation fails
        """
        # Check type
        if not isinstance(mask, np.ndarray):
            raise ValidationError("Mask must be a numpy array")
            
        # Check shape
        if len(mask.shape) != 2:
            raise ValidationError("Mask must be 2-dimensional")
            
        if mask.shape != image_shape:
            raise ValidationError(
                f"Mask shape {mask.shape} does not match image shape {image_shape}"
            )
            
        # Check values
        if mask.dtype != bool and mask.dtype != np.uint8:
            raise ValidationError("Mask must be boolean or uint8")
            
        if mask.dtype == np.uint8 and not np.all(np.isin(mask, [0, 255])):
            raise ValidationError("Uint8 mask must contain only 0 and 255 values")
            
        return mask

    @staticmethod
    def validate_features(features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted features dictionary
        Args:
            features: Features dictionary
        Returns:
            Validated features dictionary
        Raises:
            ValidationError: If validation fails
        """
        required_keys = ['colors', 'patterns', 'textures']
        
        # Check required keys
        for key in required_keys:
            if key not in features:
                raise ValidationError(f"Missing required feature: {key}")
                
        # Validate color features
        colors = features['colors']
        if not isinstance(colors, dict):
            raise ValidationError("Colors must be a dictionary")
            
        if 'dominant_colors' not in colors:
            raise ValidationError("Missing dominant colors")
            
        # Validate pattern features
        patterns = features['patterns']
        if not isinstance(patterns, dict):
            raise ValidationError("Patterns must be a dictionary")
            
        if 'pattern_type' not in patterns:
            raise ValidationError("Missing pattern type")
            
        # Validate texture features
        textures = features['textures']
        if not isinstance(textures, dict):
            raise ValidationError("Textures must be a dictionary")
            
        return features

    @staticmethod
    def validate_model_output(images: List[Image.Image], 
                            expected_size: Optional[tuple] = None) -> List[Image.Image]:
        """
        Validate model output images
        Args:
            images: List of output images
            expected_size: Optional expected image size (W, H)
        Returns:
            Validated list of images
        Raises:
            ValidationError: If validation fails
        """
        if not images:
            raise ValidationError("Empty image list")
            
        for idx, img in enumerate(images):
            if not isinstance(img, Image.Image):
                raise ValidationError(f"Invalid image at index {idx}")
                
            if expected_size and img.size != expected_size:
                raise ValidationError(
                    f"Image {idx} has wrong size: {img.size} != {expected_size}"
                )
                
        return images

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration dictionary
        Args:
            config: Configuration dictionary
        Returns:
            Validated configuration dictionary
        Raises:
            ValidationError: If validation fails
        """
        required_sections = [
            'sam_config',
            'sd_config',
            'image_config',
            'feature_config'
        ]
        
        # Check required sections
        for section in required_sections:
            if section not in config:
                raise ValidationError(f"Missing config section: {section}")
                
        # Validate SAM config
        sam_config = config['sam_config']
        if not isinstance(sam_config, dict):
            raise ValidationError("SAM config must be a dictionary")
            
        required_sam_keys = ['model_type', 'checkpoint', 'device']
        for key in required_sam_keys:
            if key not in sam_config:
                raise ValidationError(f"Missing SAM config key: {key}")
                
        # Validate SD config
        sd_config = config['sd_config']
        if not isinstance(sd_config, dict):
            raise ValidationError("SD config must be a dictionary")
            
        required_sd_keys = ['model_id', 'controlnet_model']
        for key in required_sd_keys:
            if key not in sd_config:
                raise ValidationError(f"Missing SD config key: {key}")
                
        return config

    @staticmethod
    def validate_device(device: torch.device) -> torch.device:
        """
        Validate torch device
        Args:
            device: Torch device
        Returns:
            Validated device
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(device, torch.device):
            raise ValidationError("Invalid device type")
            
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValidationError("CUDA device requested but not available")
            
        return device

    @staticmethod
    def validate_output_path(path: Union[str, Path], create_dirs: bool = True) -> str:
        """
        Validate output file path
        Args:
            path: Output file path
            create_dirs: Whether to create parent directories
        Returns:
            Validated path as string
        Raises:
            ValidationError: If validation fails
        """
        try:
            path = Path(path)
            
            # Create parent directories if needed
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
                
            # Check if parent directory exists and is writable
            if not path.parent.exists():
                raise ValidationError(f"Parent directory does not exist: {path.parent}")
                
            if not os.access(path.parent, os.W_OK):
                raise ValidationError(f"Directory not writable: {path.parent}")
                
            return str(path)
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Invalid output path: {str(e)}")
from typing import Dict, Any
import os

class Config:
    def __init__(self):
        # Base paths
        self.MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
        self.CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")

        # SAM2 Configuration
        self.SAM_CONFIG = {
            "model_type": "vit_h",  # Model type: vit_h, vit_l, vit_b
            "checkpoint": "sam_vit_h_4b8939.pth",  # Model checkpoint name
            "device": "cuda" if self._is_cuda_available() else "cpu",
            "mask_threshold": 0.5,  # Threshold for mask prediction
            "max_masks": 5  # Maximum number of masks to generate per image
        }

        # Stable Diffusion Configuration
        self.SD_CONFIG = {
            "model_id": "stabilityai/stable-diffusion-2-1",
            "controlnet_model": "lllyasviel/control-v11-p-sd15-canny",
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "strength": 0.8
        }

        # Image Processing Configuration
        self.IMAGE_CONFIG = {
            "max_size": 1024,  # Maximum image dimension
            "target_size": (768, 768),  # Target size for processing
            "formats": [".jpg", ".jpeg", ".png", ".webp"]
        }

        # Feature Extraction Configuration
        self.FEATURE_CONFIG = {
            "color_clusters": 5,  # Number of dominant colors to extract
            "pattern_threshold": 0.3,  # Threshold for pattern detection
            "texture_scale": 1.0  # Scale factor for texture analysis
        }

        # Create necessary directories
        self._create_directories()

    @staticmethod
    def _is_cuda_available() -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary"""
        return {
            "sam_config": self.SAM_CONFIG,
            "sd_config": self.SD_CONFIG,
            "image_config": self.IMAGE_CONFIG,
            "feature_config": self.FEATURE_CONFIG,
            "model_dir": self.MODEL_DIR,
            "cache_dir": self.CACHE_DIR
        }

    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values
        Args:
            updates: Dictionary containing configuration updates
        """
        for key, value in updates.items():
            if hasattr(self, key.upper() + "_CONFIG"):
                getattr(self, key.upper() + "_CONFIG").update(value)
            elif hasattr(self, key.upper()):
                setattr(self, key.upper(), value)

# Default configuration instance
config = Config()
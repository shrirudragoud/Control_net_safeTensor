import os
import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import requests
from tqdm import tqdm
import hashlib

class ModelManager:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model manager
        Args:
            config: Configuration dictionary containing model paths and settings
        """
        self.config = config
        self.model_dir = Path(config["model_dir"])
        self.cache_dir = Path(config["cache_dir"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create directories if they don't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize model cache
        self.model_cache: Dict[str, Any] = {}

    def get_model(self, model_name: str) -> Any:
        """
        Get model from cache or load it if not cached
        Args:
            model_name: Name of the model to load
        Returns:
            Loaded model
        """
        if model_name in self.model_cache:
            return self.model_cache[model_name]
        
        model = self._load_model(model_name)
        self.model_cache[model_name] = model
        return model

    def clear_cache(self, model_name: Optional[str] = None):
        """
        Clear model cache
        Args:
            model_name: Optional specific model to clear, if None clear all
        """
        if model_name is None:
            self.model_cache.clear()
            torch.cuda.empty_cache()
        elif model_name in self.model_cache:
            del self.model_cache[model_name]
            torch.cuda.empty_cache()

    def _load_model(self, model_name: str) -> Any:
        """
        Load model from disk or download if needed
        Args:
            model_name: Name of the model to load
        Returns:
            Loaded model
        """
        model_path = self.model_dir / f"{model_name}.pth"
        
        # Check if model exists locally
        if not model_path.exists():
            self._download_model(model_name, model_path)
        
        # Verify model file
        if not self._verify_model(model_path):
            self.logger.warning(f"Model verification failed for {model_name}")
            # Re-download if verification fails
            self._download_model(model_name, model_path, force=True)
        
        # Load model based on type
        if model_name.startswith("sam"):
            return self._load_sam_model(model_path)
        elif model_name.startswith("sd"):
            return self._load_sd_model(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

    def _download_model(self, model_name: str, model_path: Path, force: bool = False):
        """
        Download model from remote source
        Args:
            model_name: Name of the model to download
            model_path: Path to save the model
            force: Force download even if file exists
        """
        if model_path.exists() and not force:
            return

        # Get download URL based on model name
        url = self._get_model_url(model_name)
        
        try:
            # Download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                with open(model_path, 'wb') as f:
                    for data in response.iter_content(block_size):
                        pbar.update(len(data))
                        f.write(data)
                        
        except Exception as e:
            self.logger.error(f"Error downloading model {model_name}: {str(e)}")
            if model_path.exists():
                model_path.unlink()
            raise

    def _verify_model(self, model_path: Path) -> bool:
        """
        Verify model file integrity
        Args:
            model_path: Path to model file
        Returns:
            True if verification passes, False otherwise
        """
        try:
            # Load a small part of the model to verify it's not corrupted
            if str(model_path).endswith('.pth'):
                checkpoint = torch.load(model_path, map_location='cpu')
                if not isinstance(checkpoint, dict):
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Model verification failed: {str(e)}")
            return False

    def _get_model_url(self, model_name: str) -> str:
        """Get download URL for model"""
        # Add model URLs here
        model_urls = {
            "sam_vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            # Add other model URLs as needed
        }
        
        if model_name not in model_urls:
            raise ValueError(f"No download URL found for model: {model_name}")
            
        return model_urls[model_name]

    def _load_sam_model(self, model_path: Path) -> Any:
        """Load SAM model"""
        try:
            from segment_anything import sam_model_registry
            
            model_type = "vit_h"  # or determine from model_path
            sam = sam_model_registry[model_type](checkpoint=str(model_path))
            sam.to(device=self.device)
            return sam
        except Exception as e:
            self.logger.error(f"Error loading SAM model: {str(e)}")
            raise

    def _load_sd_model(self, model_path: Path) -> Any:
        """Load Stable Diffusion model"""
        try:
            from diffusers import StableDiffusionPipeline
            
            model = StableDiffusionPipeline.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                safety_checker=None
            )
            model.to(self.device)
            return model
        except Exception as e:
            self.logger.error(f"Error loading Stable Diffusion model: {str(e)}")
            raise

    @staticmethod
    def calculate_model_hash(model_path: Path) -> str:
        """Calculate SHA-256 hash of model file"""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def optimize_memory(self):
        """Optimize memory usage for loaded models"""
        if self.device.type == "cuda":
            # Clear unused memory
            torch.cuda.empty_cache()
            
            # Move least recently used models to CPU
            if len(self.model_cache) > 2:  # Keep only 2 models on GPU
                models_to_move = sorted(
                    self.model_cache.items(),
                    key=lambda x: x[1].last_used if hasattr(x[1], 'last_used') else 0
                )[:-2]
                
                for name, model in models_to_move:
                    model.to('cpu')
                    self.logger.info(f"Moved {name} to CPU")

    def __del__(self):
        """Cleanup on deletion"""
        self.clear_cache()
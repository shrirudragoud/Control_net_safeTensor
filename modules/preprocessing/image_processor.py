import cv2
import numpy as np
from PIL import Image, ImageOps
from typing import Tuple, Union, List
import torch
from torchvision import transforms

class ImageProcessor:
    def __init__(self, config):
        """
        Initialize the image processor with configuration
        Args:
            config: Configuration dictionary containing image processing parameters
        """
        self.config = config["image_config"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize transforms for model inputs
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def load_and_preprocess(self, image_path: str) -> Tuple[Image.Image, torch.Tensor]:
        """
        Load and preprocess an image for model input
        Args:
            image_path: Path to the input image
        Returns:
            Tuple of (PIL Image, Tensor) containing the processed image
        """
        # Load image
        image = self._load_image(image_path)
        
        # Resize image
        image = self._resize_image(image)
        
        # Convert to tensor for model input
        tensor = self.transform(image)
        
        return image, tensor.unsqueeze(0).to(self.device)

    def _load_image(self, image_path: str) -> Image.Image:
        """
        Load an image and convert to RGB
        Args:
            image_path: Path to the input image
        Returns:
            PIL Image in RGB format
        """
        # Validate file format
        if not any(image_path.lower().endswith(fmt) for fmt in self.config["formats"]):
            raise ValueError(f"Unsupported image format. Supported formats: {self.config['formats']}")
        
        # Load and convert to RGB
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)  # Handle EXIF orientation
        return image.convert('RGB')

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image while maintaining aspect ratio
        Args:
            image: Input PIL Image
        Returns:
            Resized PIL Image
        """
        width, height = image.size
        max_size = self.config["max_size"]
        
        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image

    def prepare_for_sam(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Prepare image for SAM model input
        Args:
            image: Input image (PIL Image or numpy array)
        Returns:
            Preprocessed numpy array
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        return image

    def prepare_for_sd(self, image: Image.Image) -> torch.Tensor:
        """
        Prepare image for Stable Diffusion input
        Args:
            image: Input PIL Image
        Returns:
            Normalized tensor for Stable Diffusion
        """
        # Resize to target size
        image = image.resize(self.config["target_size"], Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)

    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """
        Convert a tensor to PIL Image
        Args:
            tensor: Input tensor (C,H,W)
        Returns:
            PIL Image
        """
        tensor = tensor.squeeze().cpu()
        if tensor.dim() == 3:
            if tensor.shape[0] == 1:  # Grayscale
                tensor = tensor.repeat(3, 1, 1)
        
        # Denormalize if needed
        if tensor.min() < 0:
            tensor = torch.clamp((tensor * 0.229 + 0.485) * 255, 0, 255)
        
        return Image.fromarray(tensor.numpy().astype(np.uint8).transpose(1, 2, 0))
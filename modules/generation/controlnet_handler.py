import torch
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

class ControlNetHandler:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ControlNet handler
        Args:
            config: Configuration dictionary containing SD and ControlNet parameters
        """
        self.config = config["sd_config"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize Stable Diffusion with ControlNet"""
        # Load ControlNet model
        controlnet = ControlNetModel.from_pretrained(
            self.config["controlnet_model"],
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        )

        # Initialize pipeline
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.config["model_id"],
            controlnet=controlnet,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            safety_checker=None  # Disable safety checker for performance
        )

        # Use UNet scheduling for better quality
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )

        # Enable memory optimizations
        self.pipeline.enable_xformers_memory_efficient_attention()
        self.pipeline.enable_model_cpu_offload()

    def generate(self,
                control_image: Image.Image,
                prompt: str,
                mask: Optional[np.ndarray] = None,
                features: Optional[Dict[str, Any]] = None,
                num_images: int = 1) -> List[Image.Image]:
        """
        Generate fashion model images
        Args:
            control_image: Input clothing image for ControlNet
            prompt: Generation prompt
            mask: Optional segmentation mask
            features: Optional extracted features for prompt enhancement
            num_images: Number of images to generate
        Returns:
            List of generated images
        """
        # Enhance prompt with features if available
        if features is not None:
            prompt = self._enhance_prompt(prompt, features)

        # Prepare negative prompt
        negative_prompt = self._get_negative_prompt()

        # Generate images
        outputs = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_images_per_prompt=num_images,
            num_inference_steps=self.config["num_inference_steps"],
            guidance_scale=self.config["guidance_scale"],
            controlnet_conditioning_scale=1.0,
            generator=torch.manual_seed(42)
        )

        return self._post_process_outputs(outputs.images, mask)

    def _enhance_prompt(self, base_prompt: str, features: Dict[str, Any]) -> str:
        """Enhance generation prompt using extracted features"""
        color_info = features.get("colors", {})
        pattern_info = features.get("patterns", {})
        texture_info = features.get("textures", {})

        # Extract dominant colors
        dominant_colors = color_info.get("dominant_colors", [])
        if dominant_colors:
            # Convert RGB to color names (simplified)
            color_desc = self._get_color_description(dominant_colors[0])
        else:
            color_desc = ""

        # Get pattern information
        pattern_type = pattern_info.get("pattern_type", "solid")
        pattern_density = pattern_info.get("pattern_density", 0)

        # Get texture information
        texture_stats = texture_info.get("texture_statistics", {})
        texture_desc = self._get_texture_description(texture_stats)

        # Combine prompts
        enhanced_prompt = (
            f"{base_prompt}, "
            f"{color_desc} clothing, "
            f"{pattern_type} pattern, "
            f"{texture_desc}, "
            "professional fashion photography, highly detailed, sharp focus, "
            "fashion model, studio lighting, high-end fashion"
        )

        return enhanced_prompt

    @staticmethod
    def _get_color_description(rgb: List[int]) -> str:
        """Convert RGB to basic color description"""
        r, g, b = rgb
        # Simple color classification
        if max(r, g, b) < 50:
            return "black"
        elif min(r, g, b) > 200:
            return "white"
        elif r > g and r > b:
            return "red"
        elif g > r and g > b:
            return "green"
        elif b > r and b > g:
            return "blue"
        else:
            return "neutral-colored"

    @staticmethod
    def _get_texture_description(texture_stats: Dict[str, float]) -> str:
        """Convert texture statistics to description"""
        energy = texture_stats.get("energy", 0)
        contrast = texture_stats.get("contrast", 0)
        
        if energy > 0.8:
            return "smooth texture"
        elif contrast > 0.5:
            return "textured fabric"
        else:
            return "medium texture"

    def _get_negative_prompt(self) -> str:
        """Get negative prompt for generation"""
        return (
            "blurry, low quality, low resolution, deformed clothing, "
            "distorted proportions, bad anatomy, disfigured, poorly drawn face, "
            "extra limbs, ugly, poorly drawn hands, missing fingers, "
            "extra fingers, floating limbs, disconnected limbs, "
            "mutation, mutated, out of frame, watermark, signature, "
            "bad lighting, poorly drawn, incorrect clothing fit"
        )

    def _post_process_outputs(self,
                            images: List[Image.Image],
                            mask: Optional[np.ndarray] = None) -> List[Image.Image]:
        """
        Post-process generated images
        Args:
            images: List of generated images
            mask: Optional segmentation mask for refinement
        Returns:
            List of processed images
        """
        processed_images = []
        
        for image in images:
            # Apply mask refinement if provided
            if mask is not None:
                image = self._refine_with_mask(image, mask)
            
            # Enhance colors
            image = self._enhance_colors(image)
            
            processed_images.append(image)
        
        return processed_images

    def _refine_with_mask(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Refine generated image using the original mask"""
        # Convert images to numpy arrays
        img_array = np.array(image)
        mask_3d = np.stack([mask] * 3, axis=2)
        
        # Create blended image
        blended = img_array.copy()
        
        # Apply Gaussian blur to mask edges
        mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 2)
        mask_blur = np.stack([mask_blur] * 3, axis=2)
        
        # Blend edges
        blended = (blended * mask_blur + img_array * (1 - mask_blur)).astype(np.uint8)
        
        return Image.fromarray(blended)

    def _enhance_colors(self, image: Image.Image) -> Image.Image:
        """Enhance colors in the generated image"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(enhanced)
import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from typing import List, Tuple, Optional, Dict

class SAMHandler:
    def __init__(self, config: Dict):
        """
        Initialize SAM model handler
        Args:
            config: Configuration dictionary containing SAM model parameters
        """
        self.config = config["sam_config"]
        self.device = torch.device(self.config["device"])
        self._initialize_model()

    def _initialize_model(self):
        """Initialize SAM model and predictor"""
        sam_checkpoint = self.config["checkpoint"]
        model_type = self.config["model_type"]

        # Initialize SAM
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        
        # Create predictor
        self.predictor = SamPredictor(sam)

    def generate_masks(self, 
                      image: np.ndarray, 
                      points: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[float]]:
        """
        Generate segmentation masks for clothing in the image
        Args:
            image: Input image as numpy array (H,W,3)
            points: Optional points to guide segmentation (N,2)
        Returns:
            Tuple of (masks, scores) where masks is a numpy array of shape (N,H,W)
            and scores is a list of confidence scores
        """
        # Set image in predictor
        self.predictor.set_image(image)

        if points is None:
            # Generate automatic masks
            masks, scores, _ = self.predictor.generate_masks()
        else:
            # Generate masks based on input points
            masks, scores, _ = self.predictor.predict(
                point_coords=points,
                point_labels=np.ones(len(points)),
                multimask_output=True
            )

        # Filter masks based on threshold
        valid_masks = []
        valid_scores = []
        for mask, score in zip(masks, scores):
            if score > self.config["mask_threshold"]:
                valid_masks.append(mask)
                valid_scores.append(score)

            if len(valid_masks) >= self.config["max_masks"]:
                break

        if not valid_masks:
            raise ValueError("No valid masks found in the image")

        return np.stack(valid_masks), valid_scores

    def select_best_mask(self, masks: np.ndarray, scores: List[float]) -> np.ndarray:
        """
        Select the best mask based on clothing heuristics
        Args:
            masks: Array of masks (N,H,W)
            scores: List of confidence scores
        Returns:
            Selected mask as numpy array (H,W)
        """
        # Convert scores to numpy array
        scores = np.array(scores)

        # Calculate mask properties
        mask_properties = []
        for i, mask in enumerate(masks):
            # Calculate center of mass
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            center_y = np.mean(y_indices)
            center_x = np.mean(x_indices)
            
            # Calculate area and aspect ratio
            area = np.sum(mask)
            height = np.max(y_indices) - np.min(y_indices)
            width = np.max(x_indices) - np.min(x_indices)
            aspect_ratio = width / height if height > 0 else 0
            
            mask_properties.append({
                'index': i,
                'center_y': center_y,
                'center_x': center_x,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'score': scores[i]
            })

        if not mask_properties:
            raise ValueError("No valid masks to select from")

        # Score masks based on heuristics
        best_score = -float('inf')
        best_idx = 0

        for prop in mask_properties:
            # Prefer masks with reasonable aspect ratios (not too narrow or wide)
            aspect_score = -abs(prop['aspect_ratio'] - 0.5)
            
            # Prefer masks with larger area (but not too large)
            area_score = prop['area'] / (masks.shape[1] * masks.shape[2])
            if area_score > 0.9:  # Penalize masks that are too large
                area_score = 0
            
            # Combine scores with weights
            total_score = (
                0.4 * prop['score'] +      # SAM confidence
                0.3 * aspect_score +       # Aspect ratio
                0.3 * area_score          # Area
            )
            
            if total_score > best_score:
                best_score = total_score
                best_idx = prop['index']

        return masks[best_idx]

    def refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Refine the selected mask
        Args:
            mask: Input binary mask
        Returns:
            Refined binary mask
        """
        import cv2
        
        # Convert to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
        
        # Optional: Fill holes
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_filled = np.zeros_like(mask_cleaned)
        cv2.fillPoly(mask_filled, contours, 255)
        
        return mask_filled > 0
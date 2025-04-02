import numpy as np
import cv2
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class MaskMetrics:
    """Metrics for mask quality assessment"""
    smoothness: float
    contour_complexity: float
    area_ratio: float
    symmetry_score: float

class MaskRefiner:
    def __init__(self, config: dict):
        """
        Initialize mask refiner with configuration
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.min_mask_size = 1000  # Minimum size in pixels

    def refine(self, mask: np.ndarray) -> np.ndarray:
        """
        Refine a binary segmentation mask
        Args:
            mask: Binary mask as numpy array
        Returns:
            Refined binary mask
        """
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        # Basic cleanup
        mask = self._remove_small_regions(mask)
        
        # Smooth boundaries
        mask = self._smooth_boundaries(mask)
        
        # Fill holes
        mask = self._fill_holes(mask)
        
        # Symmetry enhancement
        mask = self._enhance_symmetry(mask)
        
        return mask > 0

    def _remove_small_regions(self, mask: np.ndarray) -> np.ndarray:
        """Remove small disconnected regions"""
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Filter components based on area
        cleaned_mask = np.zeros_like(mask)
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= self.min_mask_size:
                cleaned_mask[labels == i] = 255
                
        return cleaned_mask

    def _smooth_boundaries(self, mask: np.ndarray) -> np.ndarray:
        """Smooth mask boundaries"""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Threshold to get binary mask
        _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # Additional morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
        
        return smoothed

    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """Fill holes in the mask"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create filled mask
        filled_mask = np.zeros_like(mask)
        cv2.fillPoly(filled_mask, contours, 255)
        
        return filled_mask

    def _enhance_symmetry(self, mask: np.ndarray) -> np.ndarray:
        """Enhance mask symmetry"""
        # Find vertical axis of symmetry
        center_x = mask.shape[1] // 2
        
        # Split mask into left and right halves
        left_half = mask[:, :center_x]
        right_half = mask[:, center_x:]
        
        # Flip right half for comparison
        right_half_flipped = cv2.flip(right_half, 1)
        
        # Resize if needed
        if left_half.shape[1] != right_half_flipped.shape[1]:
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]
        
        # Combine symmetrical parts
        combined = np.zeros_like(mask)
        combined[:, :center_x] = np.maximum(left_half, right_half_flipped)
        combined[:, center_x:] = cv2.flip(combined[:, :center_x], 1)
        
        # Blend with original mask
        alpha = 0.7  # Weight for original mask
        enhanced = cv2.addWeighted(mask, alpha, combined, 1-alpha, 0)
        
        return enhanced

    def get_mask_metrics(self, mask: np.ndarray) -> MaskMetrics:
        """
        Calculate quality metrics for the mask
        Args:
            mask: Binary mask
        Returns:
            MaskMetrics object containing quality scores
        """
        # Convert to binary if needed
        binary_mask = mask > 0
        
        # Calculate smoothness (inverse of perimeter/area ratio)
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return MaskMetrics(0, 0, 0, 0)
            
        perimeter = cv2.arcLength(contours[0], True)
        area = cv2.contourArea(contours[0])
        smoothness = area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Calculate contour complexity
        approx = cv2.approxPolyDP(contours[0], 0.02 * perimeter, True)
        contour_complexity = len(approx) / perimeter if perimeter > 0 else 0
        
        # Calculate area ratio
        total_area = mask.shape[0] * mask.shape[1]
        area_ratio = area / total_area if total_area > 0 else 0
        
        # Calculate symmetry score
        center_x = mask.shape[1] // 2
        left_half = binary_mask[:, :center_x]
        right_half = cv2.flip(binary_mask[:, center_x:], 1)
        if left_half.shape == right_half.shape:
            symmetry_score = np.mean(left_half == right_half)
        else:
            symmetry_score = 0
        
        return MaskMetrics(
            smoothness=smoothness,
            contour_complexity=contour_complexity,
            area_ratio=area_ratio,
            symmetry_score=symmetry_score
        )

    def optimize_mask(self, mask: np.ndarray, 
                     target_metrics: Optional[MaskMetrics] = None) -> np.ndarray:
        """
        Optimize mask to improve quality metrics
        Args:
            mask: Input binary mask
            target_metrics: Optional target metrics to optimize towards
        Returns:
            Optimized binary mask
        """
        current_mask = mask.copy()
        current_metrics = self.get_mask_metrics(current_mask)
        
        # Define default target metrics if none provided
        if target_metrics is None:
            target_metrics = MaskMetrics(
                smoothness=0.8,
                contour_complexity=0.05,
                area_ratio=0.3,
                symmetry_score=0.9
            )
        
        # Iteratively refine mask
        iterations = 3
        for _ in range(iterations):
            # Adjust smoothness
            if current_metrics.smoothness < target_metrics.smoothness:
                current_mask = self._smooth_boundaries(current_mask)
            
            # Adjust complexity
            if current_metrics.contour_complexity > target_metrics.contour_complexity:
                kernel_size = 5
                current_mask = cv2.morphologyEx(
                    current_mask, 
                    cv2.MORPH_OPEN, 
                    np.ones((kernel_size, kernel_size), np.uint8)
                )
            
            # Enhance symmetry
            if current_metrics.symmetry_score < target_metrics.symmetry_score:
                current_mask = self._enhance_symmetry(current_mask)
            
            current_metrics = self.get_mask_metrics(current_mask)
        
        return current_mask
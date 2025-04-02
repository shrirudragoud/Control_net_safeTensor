import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

class FeatureExtractor:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature extractor
        Args:
            config: Configuration dictionary containing feature extraction parameters
        """
        self.config = config["feature_config"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_models()

    def _initialize_models(self):
        """Initialize models for feature extraction"""
        # Load pre-trained ResNet for texture analysis
        resnet = models.resnet18(pretrained=True)
        self.texture_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.texture_extractor.to(self.device)
        self.texture_extractor.eval()

        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image: Image.Image, mask: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from the clothing image
        Args:
            image: Input PIL Image
            mask: Binary mask for the clothing region
        Returns:
            Dictionary containing extracted features
        """
        # Convert mask to binary
        binary_mask = mask > 0

        # Apply mask to image
        masked_image = self._apply_mask(image, binary_mask)

        # Extract features
        color_features = self._extract_color_features(masked_image)
        pattern_features = self._extract_pattern_features(masked_image)
        texture_features = self._extract_texture_features(masked_image)

        return {
            "colors": color_features,
            "patterns": pattern_features,
            "textures": texture_features
        }

    def _apply_mask(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Apply binary mask to image"""
        # Convert PIL Image to numpy array
        image_array = np.array(image)
        
        # Create alpha channel from mask
        alpha = np.zeros_like(mask, dtype=np.uint8)
        alpha[mask] = 255
        
        # Apply mask
        masked = image_array.copy()
        masked[~mask] = 0
        
        return Image.fromarray(masked)

    def _extract_color_features(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract color features from the image
        Returns dictionary containing:
            - dominant_colors: List of RGB values
            - color_histogram: Color distribution
            - average_color: Average RGB value
        """
        # Convert to numpy array
        image_array = np.array(image)
        
        # Reshape to list of pixels
        pixels = image_array.reshape(-1, 3)
        
        # Remove background (black) pixels
        valid_pixels = pixels[~np.all(pixels == 0, axis=1)]
        
        if len(valid_pixels) == 0:
            return {
                "dominant_colors": [],
                "color_histogram": np.zeros(16),
                "average_color": np.zeros(3)
            }
        
        # Extract dominant colors using K-means
        n_colors = self.config["color_clusters"]
        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(valid_pixels)
        dominant_colors = kmeans.cluster_centers_.astype(int)
        
        # Calculate color histogram
        hist_size = 16
        color_hist = cv2.calcHist([valid_pixels], [0, 1, 2], None, 
                                 [hist_size, hist_size, hist_size], 
                                 [0, 256, 0, 256, 0, 256])
        color_hist = cv2.normalize(color_hist, color_hist).flatten()
        
        # Calculate average color
        average_color = np.mean(valid_pixels, axis=0).astype(int)
        
        return {
            "dominant_colors": dominant_colors.tolist(),
            "color_histogram": color_hist.tolist(),
            "average_color": average_color.tolist()
        }

    def _extract_pattern_features(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract pattern features from the image
        Returns dictionary containing:
            - pattern_type: Detected pattern type
            - pattern_scale: Estimated pattern scale
            - pattern_regularity: Pattern regularity score
        """
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Calculate gradients
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Calculate pattern metrics
        pattern_density = np.mean(gradient_magnitude)
        pattern_variance = np.std(gradient_magnitude)
        
        # Detect edges for pattern analysis
        edges = cv2.Canny(gray, 100, 200)
        
        # Calculate pattern regularity using Fourier transform
        f_transform = np.fft.fft2(edges)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Analyze frequency components
        center_y, center_x = magnitude_spectrum.shape[0]//2, magnitude_spectrum.shape[1]//2
        radius = 30
        circle_mask = np.zeros_like(magnitude_spectrum)
        y, x = np.ogrid[-center_y:magnitude_spectrum.shape[0]-center_y, 
                        -center_x:magnitude_spectrum.shape[1]-center_x]
        mask_area = x*x + y*y <= radius*radius
        circle_mask[mask_area] = 1
        
        # Calculate pattern regularity score
        high_freq_energy = np.sum(magnitude_spectrum * (1 - circle_mask))
        total_energy = np.sum(magnitude_spectrum)
        pattern_regularity = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # Determine pattern type
        if pattern_density < self.config["pattern_threshold"]:
            pattern_type = "solid"
        elif pattern_regularity > 0.7:
            pattern_type = "geometric"
        elif pattern_variance > 50:
            pattern_type = "complex"
        else:
            pattern_type = "simple"
        
        return {
            "pattern_type": pattern_type,
            "pattern_density": float(pattern_density),
            "pattern_regularity": float(pattern_regularity)
        }

    def _extract_texture_features(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract texture features using deep learning
        Returns dictionary containing:
            - texture_embeddings: Deep feature vector
            - texture_statistics: Basic texture statistics
        """
        # Prepare image for model
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract deep features
        with torch.no_grad():
            features = self.texture_extractor(img_tensor)
        
        # Calculate texture statistics
        features_np = features.cpu().numpy().squeeze()
        
        # Basic statistical features
        mean_features = np.mean(features_np, axis=(1, 2))
        std_features = np.std(features_np, axis=(1, 2))
        
        # Calculate GLCM features
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        glcm = self._calculate_glcm(gray)
        
        # GLCM properties
        contrast = self._glcm_contrast(glcm)
        homogeneity = self._glcm_homogeneity(glcm)
        energy = self._glcm_energy(glcm)
        correlation = self._glcm_correlation(glcm)
        
        return {
            "texture_embeddings": mean_features.tolist(),
            "texture_statistics": {
                "contrast": float(contrast),
                "homogeneity": float(homogeneity),
                "energy": float(energy),
                "correlation": float(correlation)
            }
        }

    @staticmethod
    def _calculate_glcm(image: np.ndarray) -> np.ndarray:
        """Calculate Gray-Level Co-occurrence Matrix"""
        glcm = np.zeros((256, 256))
        height, width = image.shape
        
        for i in range(height-1):
            for j in range(width-1):
                current = image[i, j]
                right = image[i, j+1]
                glcm[current, right] += 1
        
        # Normalize
        if glcm.sum() > 0:
            glcm = glcm / glcm.sum()
            
        return glcm

    @staticmethod
    def _glcm_contrast(glcm: np.ndarray) -> float:
        """Calculate GLCM contrast"""
        contrast = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                contrast += glcm[i,j] * (i-j)**2
        return contrast

    @staticmethod
    def _glcm_homogeneity(glcm: np.ndarray) -> float:
        """Calculate GLCM homogeneity"""
        homogeneity = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                homogeneity += glcm[i,j] / (1 + abs(i-j))
        return homogeneity

    @staticmethod
    def _glcm_energy(glcm: np.ndarray) -> float:
        """Calculate GLCM energy"""
        return np.sqrt(np.sum(glcm**2))

    @staticmethod
    def _glcm_correlation(glcm: np.ndarray) -> float:
        """Calculate GLCM correlation"""
        rows, cols = glcm.shape
        row_mean = np.sum(np.arange(rows) * np.sum(glcm, axis=1))
        col_mean = np.sum(np.arange(cols) * np.sum(glcm, axis=0))
        
        row_var = np.sum(((np.arange(rows) - row_mean)**2) * np.sum(glcm, axis=1))
        col_var = np.sum(((np.arange(cols) - col_mean)**2) * np.sum(glcm, axis=0))
        
        correlation = 0
        for i in range(rows):
            for j in range(cols):
                correlation += (i - row_mean) * (j - col_mean) * glcm[i,j]
        
        if row_var > 0 and col_var > 0:
            correlation /= np.sqrt(row_var * col_var)
            
        return correlation
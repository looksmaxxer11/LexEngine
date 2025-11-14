"""
Adaptive Preprocessing Module
Routes documents to optimal preprocessing based on quality assessment.
"""

from __future__ import annotations

import logging
from typing import Tuple
import numpy as np
import cv2

from .quality_assessment import assess_document_quality, classify_quality_tier

logger = logging.getLogger(__name__)


def adaptive_preprocess(image: np.ndarray, quality_tier: str = None) -> Tuple[np.ndarray, str]:
    """
    Apply adaptive preprocessing based on document quality.
    
    Quality-based routing:
    - excellent: Minimal processing (fast path)
    - good: Balanced processing (current standard)
    - medium: Enhanced processing
    - poor: Aggressive processing (maximum recovery)
    
    Args:
        image: Input image (BGR or grayscale numpy array)
        quality_tier: Pre-computed quality tier, or None to auto-assess
    
    Returns:
        Tuple of (preprocessed_image, applied_tier)
    """
    # Auto-assess quality if not provided
    if quality_tier is None:
        quality_score, _ = assess_document_quality(image)
        quality_tier = classify_quality_tier(quality_score)
    
    logger.info(f"Applying {quality_tier} preprocessing")
    
    if quality_tier == 'excellent':
        return preprocess_excellent(image), quality_tier
    elif quality_tier == 'good':
        return preprocess_good(image), quality_tier
    elif quality_tier == 'medium':
        return preprocess_medium(image), quality_tier
    else:  # poor
        return preprocess_poor(image), quality_tier


def preprocess_excellent(image: np.ndarray) -> np.ndarray:
    """
    Minimal preprocessing for high-quality documents.
    Fast path for clean digital PDFs or crisp scans.
    
    Steps:
    1. Grayscale conversion
    2. Simple Otsu binarization
    
    Processing time: ~50-100ms per page
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Simple binarization - Otsu is optimal for clean documents
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary


def preprocess_good(image: np.ndarray) -> np.ndarray:
    """
    Balanced preprocessing for good quality scans.
    Standard pipeline with moderate enhancements.
    
    Steps:
    1. Grayscale conversion
    2. CLAHE (contrast enhancement)
    3. Light denoising
    4. Otsu binarization
    5. Light morphological closing
    
    Processing time: ~200-400ms per page
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Light denoising (faster than fastNlMeansDenoising)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Binarization
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Light morphological closing to connect broken characters
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return closed


def preprocess_medium(image: np.ndarray) -> np.ndarray:
    """
    Enhanced preprocessing for medium quality scans.
    More aggressive filtering for noisy or faded documents.
    
    Steps:
    1. Grayscale conversion
    2. Bilateral filter (edge-preserving denoising)
    3. CLAHE with higher clip limit
    4. Adaptive thresholding
    5. Morphological operations
    
    Processing time: ~400-800ms per page
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Bilateral filter - preserves edges while removing noise
    bilateral = cv2.bilateralFilter(gray, 5, 50, 50)
    
    # CLAHE with higher clip limit for faded documents
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(bilateral)
    
    # Adaptive thresholding (better for uneven lighting)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, 8
    )
    
    # Morphological closing to connect broken characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return closed


def preprocess_poor(image: np.ndarray) -> np.ndarray:
    """
    Aggressive preprocessing for poor quality scans.
    Maximum recovery for noisy, faded, or damaged documents.
    
    Steps:
    1. Grayscale conversion
    2. Bilateral filter (stronger)
    3. CLAHE (aggressive)
    4. Adaptive thresholding
    5. Morphological operations (closing + dilation)
    6. Optional: Deskewing
    
    Processing time: ~800-1500ms per page
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Strong bilateral filter for heavy noise
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Aggressive CLAHE for very faded documents
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bilateral)
    
    # Adaptive thresholding with larger window
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 21, 10
    )
    
    # Morphological closing to connect broken characters
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # Dilation to thicken thin strokes
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilated = cv2.dilate(closed, kernel_dilate, iterations=1)
    
    # Optional: Deskewing (expensive but helps for severely tilted pages)
    try:
        dilated = _light_deskew(dilated)
    except Exception as e:
        logger.debug(f"Deskewing failed (non-critical): {e}")
    
    return dilated


def _light_deskew(binary: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
    """
    Light deskewing for slightly tilted documents.
    Only corrects angles up to max_angle degrees to avoid over-rotation.
    """
    # Find text pixels
    coords = np.column_stack(np.where(binary == 0))
    
    if coords.size == 0:
        return binary
    
    # Calculate rotation angle via minimum area rectangle
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    
    # Normalize angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Only apply small corrections
    if abs(angle) > max_angle:
        logger.debug(f"Skipping deskew: angle {angle:.1f}° exceeds threshold {max_angle}°")
        return binary
    
    # Rotate image
    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    logger.debug(f"Applied deskew: {angle:.2f}°")
    return rotated


__all__ = [
    'adaptive_preprocess',
    'preprocess_excellent',
    'preprocess_good',
    'preprocess_medium',
    'preprocess_poor',
]

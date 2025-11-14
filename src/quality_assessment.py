"""
Document Quality Assessment Module
Classifies document quality to route to appropriate preprocessing pipelines.
"""

from __future__ import annotations

import logging
from typing import Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def assess_document_quality(image: np.ndarray) -> Tuple[float, dict]:
    """
    Assess document quality and return score + detailed metrics.
    
    Returns quality score 0.0-1.0 based on:
    - Edge sharpness (Laplacian variance) 
    - Contrast ratio
    - Text density estimation
    - Noise level (standard deviation in uniform regions)
    
    Args:
        image: Input image (BGR or grayscale numpy array)
    
    Returns:
        Tuple of (quality_score, metrics_dict)
        - quality_score: 0.0 (worst) to 1.0 (best)
        - metrics_dict: Detailed breakdown of quality factors
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 1. Edge Sharpness (Laplacian variance)
    # Higher variance = sharper edges = better quality
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = laplacian.var()
    sharpness_raw = laplacian_var
    sharpness = min(laplacian_var / 1000.0, 1.0)  # Normalize (1000 is empirical threshold)
    
    # 2. Contrast Ratio
    # Higher std dev = better contrast
    contrast_raw = gray.std()
    contrast = min(contrast_raw / 70.0, 1.0)  # Normalize (70 is good contrast)
    
    # 3. Text Density (estimate via edge detection)
    edges = cv2.Canny(gray, 50, 150)
    text_density_raw = (edges > 0).sum() / edges.size
    text_density = min(text_density_raw * 10, 1.0)  # Normalize
    
    # 4. Noise Level (inverse - lower is better)
    # Compare original vs slightly blurred to detect noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_raw = np.abs(gray.astype(float) - blur.astype(float)).mean()
    noise_score = 1.0 - min(noise_raw / 20.0, 1.0)  # Inverse (20 is high noise threshold)
    
    # 5. Dynamic Range
    # Good scans use full range [0, 255]
    min_val = gray.min()
    max_val = gray.max()
    dynamic_range_raw = max_val - min_val
    dynamic_range = dynamic_range_raw / 255.0
    
    # Combined Quality Score (weighted average)
    weights = {
        'sharpness': 0.30,
        'contrast': 0.25,
        'text_density': 0.15,
        'noise_score': 0.20,
        'dynamic_range': 0.10,
    }
    
    quality = (
        sharpness * weights['sharpness'] +
        contrast * weights['contrast'] +
        text_density * weights['text_density'] +
        noise_score * weights['noise_score'] +
        dynamic_range * weights['dynamic_range']
    )
    
    quality = np.clip(quality, 0.0, 1.0)
    
    # Detailed metrics for logging/debugging
    metrics = {
        'quality_score': float(quality),
        'sharpness': float(sharpness),
        'sharpness_raw': float(sharpness_raw),
        'contrast': float(contrast),
        'contrast_raw': float(contrast_raw),
        'text_density': float(text_density),
        'text_density_raw': float(text_density_raw),
        'noise_score': float(noise_score),
        'noise_raw': float(noise_raw),
        'dynamic_range': float(dynamic_range),
        'dynamic_range_raw': float(dynamic_range_raw),
    }
    
    logger.info(f"Quality assessment: {quality:.3f} (sharpness={sharpness:.2f}, contrast={contrast:.2f}, noise={noise_score:.2f})")
    
    return quality, metrics


def classify_quality_tier(quality_score: float) -> str:
    """
    Classify document into quality tiers for routing.
    
    Tiers:
    - 'excellent': 0.80+ (clean digital PDFs, minimal preprocessing)
    - 'good': 0.60-0.80 (good scans, standard preprocessing)
    - 'medium': 0.40-0.60 (average scans, enhanced preprocessing)
    - 'poor': <0.40 (noisy/faded scans, aggressive preprocessing)
    
    Args:
        quality_score: Quality score from assess_document_quality()
    
    Returns:
        Quality tier string
    """
    if quality_score >= 0.80:
        return 'excellent'
    elif quality_score >= 0.60:
        return 'good'
    elif quality_score >= 0.40:
        return 'medium'
    else:
        return 'poor'


def calculate_optimal_dpi(pdf_path: str, sample_pages: int = 1) -> int:
    """
    Analyze PDF to determine optimal rendering DPI based on content.
    
    Higher DPI for:
    - Small text (newspapers often 7-9pt)
    - Dense layouts
    - Low-resolution source scans
    
    Lower DPI for:
    - Large text (>12pt)
    - Simple layouts
    - Already high-quality digital PDFs
    
    Args:
        pdf_path: Path to PDF file
        sample_pages: Number of pages to sample (default: 1)
    
    Returns:
        Optimal DPI (300, 400, or 600)
    """
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF not available, using default 400 DPI")
        return 400
    
    try:
        doc = fitz.open(pdf_path)
        
        if len(doc) == 0:
            doc.close()
            return 400
        
        # Sample first N pages
        pages_to_check = min(sample_pages, len(doc))
        font_sizes = []
        has_images = False
        
        for page_num in range(pages_to_check):
            page = doc[page_num]
            
            # Check for embedded text (digital PDF vs scanned)
            text_dict = page.get_text("dict")
            blocks = text_dict.get("blocks", [])
            
            # Collect font sizes
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            size = span.get("size", 0)
                            if size > 0:
                                font_sizes.append(size)
            
            # Check for images (scanned pages)
            images = page.get_images(full=True)
            if images:
                has_images = True
        
        doc.close()
        
        # Decision logic
        if not font_sizes:
            # No embedded text = scanned image, use high DPI
            logger.info("No embedded text detected (scanned document), using 600 DPI")
            return 600
        
        # Calculate average font size
        avg_font_size = sum(font_sizes) / len(font_sizes)
        median_font_size = sorted(font_sizes)[len(font_sizes) // 2]
        
        logger.info(f"Font analysis: avg={avg_font_size:.1f}pt, median={median_font_size:.1f}pt, has_images={has_images}")
        
        # DPI selection based on text size
        if median_font_size >= 12:
            # Large text (reports, presentations)
            dpi = 300
        elif median_font_size >= 9:
            # Medium text (standard documents)
            dpi = 400
        else:
            # Small text (newspapers, dense layouts)
            dpi = 600
        
        # Boost DPI if document has images (likely scanned)
        if has_images and dpi < 400:
            dpi = 400
            logger.info(f"Boosting DPI to {dpi} due to embedded images")
        
        logger.info(f"Selected DPI: {dpi} (avg_font={avg_font_size:.1f}pt)")
        return dpi
        
    except Exception as e:
        logger.error(f"Error analyzing PDF for DPI selection: {e}")
        return 400  # Safe default


__all__ = [
    'assess_document_quality',
    'classify_quality_tier',
    'calculate_optimal_dpi',
]

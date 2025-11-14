"""
Advanced PDF preprocessing with line detection, announcement segmentation, and quality preservation.
Implements Phase 1 of enhanced preprocessing pipeline.
"""

from __future__ import annotations

import os
from typing import List, Tuple, Optional
import numpy as np
import cv2
from pathlib import Path


def pdf_to_images(pdf_path: str, dpi: int = None) -> List[np.ndarray]:
    """
    Converts a PDF to high-quality images preserving quality.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution in DPI (None=auto-detect, or manual: 300-600)
    
    Returns:
        List of numpy arrays (BGR) for each page
    """
    # Normalize path to handle Windows paths with spaces
    import os
    pdf_path = os.path.normpath(os.path.abspath(pdf_path))
    
    # Verify file exists before attempting to open
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Auto-detect optimal DPI if not specified
    if dpi is None:
        from .quality_assessment import calculate_optimal_dpi
        dpi = calculate_optimal_dpi(pdf_path)
        import logging
        logging.info(f"Auto-selected DPI: {dpi} for {os.path.basename(pdf_path)}")
    
    # Try pdf2image first (requires Poppler)
    try:
        from pdf2image import convert_from_path
        pages_pil = convert_from_path(pdf_path, dpi=dpi, fmt='png')
        # Convert PIL images to numpy arrays
        return [np.array(page) for page in pages_pil]
    except Exception:
        pass
    
    # Fallback to PyMuPDF (fitz) - no external dependencies
    try:
        import fitz
        doc = fitz.open(pdf_path)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        images: List[np.ndarray] = []
        
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            images.append(img)
        
        doc.close()
        return images
    except Exception as e:
        raise RuntimeError(
            "Could not convert PDF to images. Install either:\n"
            "  pip install pdf2image (requires Poppler)\n"
            "  pip install pymupdf (recommended for Windows)"
        ) from e


def detect_horizontal_lines(image: np.ndarray, min_length: int = 100) -> List[Tuple[int, int, int, int]]:
    """
    Detects horizontal lines in an image using OpenCV edge detection and Hough transform.
    These lines typically separate announcements in newspaper scans.
    
    Args:
        image: Input image (numpy array, BGR or grayscale)
        min_length: Minimum line length in pixels
    
    Returns:
        List of line coordinates [(x1, y1, x2, y2), ...]
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=min_length,
        maxLineGap=10
    )
    
    if lines is None:
        return []
    
    # Filter for horizontal lines (angle close to 0 or 180 degrees)
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate angle
        if x2 - x1 != 0:
            angle = abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
            # Keep lines that are nearly horizontal (within 10 degrees)
            if angle < 10 or angle > 170:
                horizontal_lines.append((x1, y1, x2, y2))
        else:
            # Vertical line, skip
            continue
    
    return horizontal_lines


def merge_nearby_lines(lines: List[Tuple[int, int, int, int]], threshold: int = 20) -> List[int]:
    """
    Merges nearby horizontal lines and returns y-coordinates of separator lines.
    
    Args:
        lines: List of line coordinates [(x1, y1, x2, y2), ...]
        threshold: Maximum distance in pixels to merge lines
    
    Returns:
        List of y-coordinates for separator lines (sorted)
    """
    if not lines:
        return []
    
    # Get average y-coordinate for each line
    y_coords = [(y1 + y2) // 2 for x1, y1, x2, y2 in lines]
    y_coords = sorted(set(y_coords))
    
    # Merge nearby lines
    merged = []
    current = y_coords[0]
    
    for y in y_coords[1:]:
        if y - current <= threshold:
            current = (current + y) // 2  # Average
        else:
            merged.append(current)
            current = y
    merged.append(current)
    
    return merged


def crop_by_separator_lines(image: np.ndarray, y_coords: List[int], margin: int = 10) -> List[np.ndarray]:
    """
    Crops an image into regions based on separator line y-coordinates.
    
    Args:
        image: Input image (numpy array)
        y_coords: List of y-coordinates for separator lines (sorted)
        margin: Margin in pixels to add around each crop
    
    Returns:
        List of cropped image regions
    """
    if not y_coords:
        return [image]
    
    height, width = image.shape[:2]
    cropped_regions = []
    
    # Add boundaries at top and bottom
    y_coords = [0] + y_coords + [height]
    
    for i in range(len(y_coords) - 1):
        y1 = max(0, y_coords[i] - margin)
        y2 = min(height, y_coords[i + 1] + margin)
        
        # Only keep regions with sufficient height
        if y2 - y1 > 50:  # Minimum 50 pixels height
            cropped_regions.append(image[y1:y2, :])
    
    return cropped_regions


def enhance_image_for_ocr(image: np.ndarray, zoom_scale: float = 1.5, quality_tier: str = None) -> np.ndarray:
    """
    Enhances an image for better OCR using adaptive preprocessing.
    
    Now routes to quality-based preprocessing pipelines:
    - excellent: minimal processing (fast)
    - good: balanced processing
    - medium: enhanced processing
    - poor: aggressive processing
    
    Args:
        image: Input image (numpy array)
        zoom_scale: Scale factor for zooming (applied after preprocessing)
        quality_tier: Pre-assessed quality tier (None=auto-assess)
    
    Returns:
        Enhanced image (numpy array)
    """
    from .adaptive_preprocessing import adaptive_preprocess
    
    # Apply adaptive preprocessing based on quality
    preprocessed, applied_tier = adaptive_preprocess(image, quality_tier)
    
    # Apply zoom if requested (zoom after preprocessing for better results)
    if zoom_scale != 1.0 and zoom_scale > 0:
        height, width = preprocessed.shape[:2]
        new_width = int(width * zoom_scale)
        new_height = int(height * zoom_scale)
        zoomed = cv2.resize(preprocessed, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return zoomed
    
    return preprocessed


def process_pdf_with_line_detection(
    pdf_path: str,
    output_dir: str,
    dpi: int = None,
    zoom_scale: float = 1.5,
    detect_lines: bool = True
) -> List[str]:
    """
    Full pipeline for processing PDF with line detection and announcement segmentation.
    Now includes adaptive DPI selection and quality-based preprocessing routing.
    
    Args:
        pdf_path: Path to input PDF
        output_dir: Directory to save processed images
        dpi: Resolution for PDF-to-image conversion (None=auto-detect optimal)
        zoom_scale: Zoom factor for enhanced regions
        detect_lines: Whether to detect separator lines
    
    Returns:
        List of file paths to processed image segments
    """
    import logging
    from .quality_assessment import assess_document_quality, classify_quality_tier
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert PDF to high-quality images (with adaptive DPI)
    pages = pdf_to_images(pdf_path, dpi=dpi)
    
    # Assess quality of first page to determine preprocessing strategy
    if pages:
        quality_score, metrics = assess_document_quality(pages[0])
        quality_tier = classify_quality_tier(quality_score)
        logging.info(f"Document quality: {quality_score:.3f} ({quality_tier}) - {metrics}")
    else:
        quality_tier = 'good'  # Default fallback
    
    output_paths = []
    segment_counter = 0
    
    for page_idx, page_image in enumerate(pages):
        if detect_lines:
            # Detect horizontal separator lines
            lines = detect_horizontal_lines(page_image, min_length=200)
            y_coords = merge_nearby_lines(lines, threshold=30)
            
            # Crop into announcement regions
            regions = crop_by_separator_lines(page_image, y_coords, margin=10)
        else:
            # Process entire page as single region
            regions = [page_image]
        
        # Process each region with quality-aware preprocessing
        for region_idx, region in enumerate(regions):
            # Enhance for OCR using adaptive preprocessing
            enhanced = enhance_image_for_ocr(region, zoom_scale=zoom_scale, quality_tier=quality_tier)
            
            # Save processed region
            output_path = os.path.join(
                output_dir,
                f"segment_{segment_counter:04d}_page{page_idx:02d}_region{region_idx:02d}.png"
            )
            cv2.imwrite(output_path, enhanced)
            output_paths.append(os.path.abspath(output_path))
            segment_counter += 1
    
    return output_paths


def adaptive_script_detection(text_sample: str) -> str:
    """
    Detects dominant script in a text sample and returns optimal Tesseract language string.
    
    Args:
        text_sample: Sample text to analyze
    
    Returns:
        Optimal language string for Tesseract (e.g., 'rus+uzb', 'uzb+eng')
    """
    if not text_sample:
        return "uzb+rus+eng"
    
    # Count Cyrillic vs Latin characters
    cyrillic_count = sum(1 for c in text_sample if '\u0400' <= c <= '\u04FF')
    latin_count = sum(1 for c in text_sample if ('a' <= c.lower() <= 'z'))
    total = cyrillic_count + latin_count
    
    if total == 0:
        return "uzb+rus+eng"
    
    cyrillic_ratio = cyrillic_count / total
    
    if cyrillic_ratio > 0.7:
        # Predominantly Cyrillic (Russian/Uzbek Cyrillic)
        return "rus+uzb"
    elif cyrillic_ratio < 0.3:
        # Predominantly Latin (English/Uzbek Latin)
        return "uzb+eng"
    else:
        # Mixed scripts
        return "uzb+rus+eng"


__all__ = [
    "pdf_to_images",
    "detect_horizontal_lines",
    "merge_nearby_lines",
    "crop_by_separator_lines",
    "enhance_image_for_ocr",
    "process_pdf_with_line_detection",
    "adaptive_script_detection",
]

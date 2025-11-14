"""
Phase 3.5: Region-Based OCR with Reading Order Correction
Processes document regions in correct top-to-bottom, left-to-right order
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


@dataclass
class TextRegion:
    """Represents a text region with spatial information."""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    text: str = ""
    confidence: float = 0.0
    column_index: int = 0
    reading_order: int = 0
    region_type: str = "body"  # body, header, footer, table


class ColumnDetector:
    """Detects vertical columns in document images."""
    
    def __init__(self, min_column_gap: int = 30, min_column_width: int = 100):
        self.min_column_gap = min_column_gap
        self.min_column_width = min_column_width
    
    def detect_columns(self, image: np.ndarray, text_regions: List[TextRegion]) -> List[Dict]:
        """
        Detect column boundaries by analyzing X-position distribution of regions.
        Uses clustering to identify natural column boundaries in newspaper layouts.
        
        Returns list of column dicts: {'x_start': int, 'x_end': int, 'index': int}
        """
        if not text_regions:
            h, w = image.shape[:2]
            return [{'x_start': 0, 'x_end': w, 'index': 0}]
        
        img_height, img_width = image.shape[:2]
        
        # Collect region left and right edges
        x_starts = sorted([r.bbox[0] for r in text_regions])
        x_ends = sorted([r.bbox[0] + r.bbox[2] for r in text_regions])
        
        # Find vertical whitespace by looking for large gaps
        # Look for gaps that are consistently present across many Y-levels
        vertical_gaps = []
        
        # Sample vertical positions
        for sample_y in range(0, img_height, img_height // 10):
            # Find regions at this Y level
            regions_at_y = [r for r in text_regions if r.bbox[1] <= sample_y <= r.bbox[1] + r.bbox[3]]
            
            if len(regions_at_y) < 2:
                continue
            
            # Sort by X position
            regions_at_y.sort(key=lambda r: r.bbox[0])
            
            # Find gaps between adjacent regions
            for i in range(len(regions_at_y) - 1):
                curr_end = regions_at_y[i].bbox[0] + regions_at_y[i].bbox[2]
                next_start = regions_at_y[i + 1].bbox[0]
                gap = next_start - curr_end
                
                if gap >= self.min_column_gap:
                    gap_center = (curr_end + next_start) // 2
                    vertical_gaps.append(gap_center)
        
        if not vertical_gaps:
            logging.info("No consistent column gaps - single column")
            return [{'x_start': 0, 'x_end': img_width, 'index': 0}]
        
        # Cluster gap positions (gaps at similar X positions across Y levels = column boundary)
        vertical_gaps.sort()
        column_boundaries = []
        
        if vertical_gaps:
            current_cluster = [vertical_gaps[0]]
            
            for gap_x in vertical_gaps[1:]:
                if gap_x - current_cluster[-1] < 100:  # Close gaps belong to same boundary
                    current_cluster.append(gap_x)
                else:
                    # Finalize this boundary (use median)
                    if len(current_cluster) >= 3:  # Require at least 3 samples
                        boundary_x = int(np.median(current_cluster))
                        column_boundaries.append(boundary_x)
                    current_cluster = [gap_x]
            
            # Final cluster
            if len(current_cluster) >= 3:
                boundary_x = int(np.median(current_cluster))
                column_boundaries.append(boundary_x)
        
        if not column_boundaries:
            logging.info("No reliable column boundaries - single column")
            return [{'x_start': 0, 'x_end': img_width, 'index': 0}]
        
        # Build columns from boundaries
        columns = []
        col_start = 0
        
        for boundary in column_boundaries:
            columns.append({
                'x_start': col_start,
                'x_end': boundary,
                'index': len(columns)
            })
            col_start = boundary
        
        # Final column
        columns.append({
            'x_start': col_start,
            'x_end': img_width,
            'index': len(columns)
        })
        
        logging.info(f"✅ Detected {len(columns)} column(s) at boundaries: {column_boundaries}")
        return columns
    
    def assign_regions_to_columns(self, regions: List[TextRegion], columns: List[Dict]) -> None:
        """Assign each region to its corresponding column."""
        for region in regions:
            x, y, w, h = region.bbox
            region_center_x = x + w // 2
            
            # Find which column this region belongs to
            for col in columns:
                if col['x_start'] <= region_center_x <= col['x_end']:
                    region.column_index = col['index']
                    break


class ReadingOrderSorter:
    """Sorts text regions into natural reading order."""
    
    def __init__(self, header_margin: int = 100, footer_margin: int = 100):
        self.header_margin = header_margin
        self.footer_margin = footer_margin
    
    def classify_region_type(self, region: TextRegion, img_height: int) -> str:
        """Classify region as header, footer, or body."""
        x, y, w, h = region.bbox
        
        if y < self.header_margin:
            return "header"
        elif y + h > img_height - self.footer_margin:
            return "footer"
        else:
            return "body"
    
    def sort_regions(self, regions: List[TextRegion], img_height: int) -> List[TextRegion]:
        """
        Sort regions in natural reading order:
        1. Headers (top to bottom)
        2. Body text by columns (left-to-right), then top-to-bottom within column
        3. Footers (top to bottom)
        """
        # Classify regions
        headers = []
        body = []
        footers = []
        
        for region in regions:
            region_type = self.classify_region_type(region, img_height)
            region.region_type = region_type
            
            if region_type == "header":
                headers.append(region)
            elif region_type == "footer":
                footers.append(region)
            else:
                body.append(region)
        
        # Sort headers top-to-bottom
        headers.sort(key=lambda r: r.bbox[1])
        
        # Sort body by column (left-to-right), then y (top-to-bottom) within column
        body.sort(key=lambda r: (r.column_index, r.bbox[1]))
        
        # Sort footers top-to-bottom
        footers.sort(key=lambda r: r.bbox[1])
        
        # Combine in reading order
        ordered_regions = headers + body + footers
        
        # Assign final reading order numbers
        for idx, region in enumerate(ordered_regions):
            region.reading_order = idx
        
        return ordered_regions


class RegionOCR:
    """
    Performs OCR on document regions in correct reading order.
    Ensures top-to-bottom, left-to-right processing for multi-column layouts.
    """
    
    def __init__(self, ocr_engine, min_column_gap: int = 30, max_workers: int = 4):
        self.ocr_engine = ocr_engine
        self.column_detector = ColumnDetector(min_column_gap=min_column_gap)
        self.reading_sorter = ReadingOrderSorter()
        self.max_workers = max_workers  # Parallel OCR threads
    
    def extract_text_regions(self, image: np.ndarray) -> List[TextRegion]:
        """
        Extract text regions from image using contour detection.
        Returns list of TextRegion objects with bounding boxes.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to connect text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        img_height, img_width = image.shape[:2]
        
        # Adaptive filtering based on image size
        # Newspapers (large images) need gentler filtering to capture small text
        # Simple documents can use stricter filtering for speed
        if img_width > 2000 or img_height > 2000:
            # Newspaper/large document mode - keep more regions
            min_area = img_width * img_height * 0.00005  # 0.005% of image
            min_w, min_h = 15, 10  # Smaller threshold for newspaper text
            logging.info("📰 Large document detected - using newspaper-optimized filtering")
        else:
            # Simple/small document mode - filter more aggressively
            min_area = img_width * img_height * 0.0005  # 0.05% of image
            min_w, min_h = 30, 15  # Stricter for speed
            logging.info("📄 Standard document detected - using strict filtering")
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter out tiny regions with adaptive thresholds
            if area < min_area or w < min_w or h < min_h:
                continue
            
            region = TextRegion(bbox=(x, y, w, h))
            regions.append(region)
        
        logging.info(f"Extracted {len(regions)} text region(s) (filtered regions < {min_w}x{min_h}px, {min_area:.0f}px² area)")
        return regions
    
    def process_image_with_reading_order(
        self, 
        image: np.ndarray,
        use_multiscale: bool = False,
        use_retry: bool = False
    ) -> Tuple[str, List[TextRegion]]:
        """
        Main entry point: Process image with correct reading order.
        
        Returns:
            - Combined text in reading order
            - List of processed TextRegion objects
        """
        img_height, img_width = image.shape[:2]
        
        # Step 1: Extract text regions
        regions = self.extract_text_regions(image)
        
        if not regions:
            logging.warning("No text regions detected")
            return "", []
        
        # Step 2: Detect columns
        columns = self.column_detector.detect_columns(image, regions)
        
        # Step 3: Assign regions to columns
        self.column_detector.assign_regions_to_columns(regions, columns)
        
        # Step 4: Sort regions in reading order
        ordered_regions = self.reading_sorter.sort_regions(regions, img_height)
        
        # Step 5: OCR each region with progress logging
        start_time = time.time()
        logging.info(f"🔄 Starting OCR on {len(ordered_regions)} regions...")
        
        for idx, region in enumerate(ordered_regions, 1):
            x, y, w, h = region.bbox
            region_img = image[y:y+h, x:x+w]
            
            # Log progress every 5 regions
            if idx % 5 == 0 or idx == 1:
                logging.info(f"  Processing region {idx}/{len(ordered_regions)}...")
            
            # Perform OCR on this region
            if hasattr(self.ocr_engine, 'ocr_image'):
                result = self.ocr_engine.ocr_image(region_img)
                region.text = result.get('text', '')
                region.confidence = result.get('confidence', 0.0)
            else:
                # Fallback for simple OCR engines
                import pytesseract
                ocr_result = pytesseract.image_to_data(
                    region_img, 
                    output_type=pytesseract.Output.DICT,
                    lang='rus+uzb',
                    config='--psm 6 --oem 3'  # Optimized config
                )
                text_parts = [
                    word for word, conf in zip(ocr_result['text'], ocr_result['conf']) 
                    if conf > 30  # Confidence threshold
                ]
                region.text = ' '.join(text_parts)
                region.confidence = np.mean([c for c in ocr_result['conf'] if c > 30]) if text_parts else 0.0
        
        ocr_time = time.time() - start_time
        logging.info(f"⚡ OCR completed in {ocr_time:.1f}s ({len(ordered_regions)} regions, {ocr_time/len(ordered_regions):.2f}s per region)")
        
        # Step 6: Combine text in reading order
        full_text_parts = []
        
        for idx, region in enumerate(ordered_regions):
            # Skip empty regions
            if not region.text.strip():
                continue
            
            # Add column break for new columns
            if idx > 0 and region.column_index > ordered_regions[idx-1].column_index:
                full_text_parts.append("\n\n")  # Double newline for column break
            
            full_text_parts.append(region.text.strip())
            full_text_parts.append("\n")  # Single newline within column
        
        full_text = ''.join(full_text_parts)
        
        logging.info(f"Processed {len(ordered_regions)} regions in reading order")
        logging.info(f"Total text length: {len(full_text)} characters")
        logging.info(f"✅ RegionOCR: Processed {len(ordered_regions)} regions in CORRECT READING ORDER (top-to-bottom)")
        
        return full_text, ordered_regions
        
        full_text = ''.join(full_text_parts)
        
        logging.info(f"Processed {len(ordered_regions)} regions in reading order")
        logging.info(f"Total text length: {len(full_text)} characters")
        
        return full_text, ordered_regions
    
    def visualize_reading_order(self, image: np.ndarray, regions: List[TextRegion], output_path: str) -> None:
        """
        Create visualization showing reading order with numbered regions.
        """
        vis_img = image.copy()
        
        # Define colors for different region types
        colors = {
            'header': (0, 0, 255),    # Red
            'body': (0, 255, 0),      # Green
            'footer': (255, 0, 0)     # Blue
        }
        
        for region in regions:
            x, y, w, h = region.bbox
            color = colors.get(region.region_type, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)
            
            # Draw reading order number
            cv2.putText(
                vis_img,
                str(region.reading_order),
                (x + 5, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )
            
            # Draw column index
            cv2.putText(
                vis_img,
                f"C{region.column_index}",
                (x + 5, y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        cv2.imwrite(output_path, vis_img)
        logging.info(f"Reading order visualization saved to {output_path}")


def process_with_reading_order(image_path: str, ocr_engine, output_path: str = None) -> str:
    """
    Convenience function to process an image with reading order correction.
    
    Args:
        image_path: Path to input image
        ocr_engine: OCR engine instance
        output_path: Optional path to save visualization
    
    Returns:
        Extracted text in correct reading order
    """
    import cv2
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Create RegionOCR processor
    region_ocr = RegionOCR(ocr_engine)
    
    # Process image
    text, regions = region_ocr.process_image_with_reading_order(image)
    
    # Save visualization if requested
    if output_path:
        region_ocr.visualize_reading_order(image, regions, output_path)
    
    return text

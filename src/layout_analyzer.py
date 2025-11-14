"""
Phase 3: Advanced Layout Analysis
Intelligent document structure recognition and reading order optimization.
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2


class RegionType(Enum):
    """Types of document regions."""
    TEXT = "text"
    TITLE = "title"
    HEADING = "heading"
    TABLE = "table"
    IMAGE = "image"
    FOOTER = "footer"
    HEADER = "header"
    COLUMN = "column"
    UNKNOWN = "unknown"


@dataclass
class LayoutRegion:
    """Represents a region in the document with semantic information."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    region_type: RegionType
    confidence: float
    text: str = ""
    reading_order: int = -1
    column_id: int = -1
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def area(self) -> int:
        """Calculate region area."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    @property
    def center(self) -> Tuple[int, int]:
        """Calculate region center point."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class LayoutAnalyzer:
    """
    Advanced layout analysis for document structure recognition.
    
    Features:
    - Multi-column detection
    - Reading order determination
    - Table structure recognition
    - Header/footer identification
    - Semantic region classification
    """
    
    def __init__(self):
        self.min_column_gap = 30  # Minimum gap between columns (pixels)
        self.header_footer_margin = 100  # Margin for header/footer detection
        
    def analyze_layout(self, image_path: str) -> List[LayoutRegion]:
        """
        Perform comprehensive layout analysis.
        
        Args:
            image_path: Path to document image
            
        Returns:
            List of LayoutRegion objects with semantic information
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"Failed to load image: {image_path}")
            return []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Detect regions using contours
        regions = self._detect_regions(gray)
        
        # Classify region types
        regions = self._classify_regions(regions, height, width)
        
        # Detect columns
        regions = self._detect_columns(regions, width)
        
        # Determine reading order
        regions = self._determine_reading_order(regions)
        
        logging.info(f"Layout analysis complete: {len(regions)} regions detected")
        return regions
    
    def _detect_regions(self, gray: np.ndarray) -> List[LayoutRegion]:
        """Detect text regions using morphological operations."""
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to connect text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small regions
            if w < 50 or h < 20:
                continue
            
            bbox = (x, y, x + w, y + h)
            region = LayoutRegion(
                bbox=bbox,
                region_type=RegionType.TEXT,
                confidence=1.0
            )
            regions.append(region)
        
        return regions
    
    def _classify_regions(
        self, 
        regions: List[LayoutRegion], 
        page_height: int, 
        page_width: int
    ) -> List[LayoutRegion]:
        """Classify regions by type (header, footer, title, text, etc.)."""
        for region in regions:
            x1, y1, x2, y2 = region.bbox
            height = y2 - y1
            width = x2 - x1
            aspect_ratio = width / height if height > 0 else 0
            
            # Header detection (top of page)
            if y1 < self.header_footer_margin:
                region.region_type = RegionType.HEADER
                region.metadata['position'] = 'header'
            
            # Footer detection (bottom of page)
            elif y2 > page_height - self.header_footer_margin:
                region.region_type = RegionType.FOOTER
                region.metadata['position'] = 'footer'
            
            # Title detection (large, wide region near top)
            elif y1 < page_height * 0.2 and width > page_width * 0.6 and height > 40:
                region.region_type = RegionType.TITLE
                region.metadata['position'] = 'title'
            
            # Heading detection (wide region, medium height)
            elif width > page_width * 0.5 and 25 < height < 60:
                region.region_type = RegionType.HEADING
                region.metadata['position'] = 'heading'
            
            # Table detection (large, nearly square aspect ratio)
            elif 0.5 < aspect_ratio < 2.0 and width > 200 and height > 100:
                region.region_type = RegionType.TABLE
                region.metadata['position'] = 'table'
            
            # Regular text
            else:
                region.region_type = RegionType.TEXT
                region.metadata['position'] = 'body'
        
        return regions
    
    def _detect_columns(self, regions: List[LayoutRegion], page_width: int) -> List[LayoutRegion]:
        """Detect column structure in the document."""
        # Filter body text regions
        text_regions = [r for r in regions if r.region_type == RegionType.TEXT]
        
        if not text_regions:
            return regions
        
        # Get X coordinates of region centers
        x_centers = [r.center[0] for r in text_regions]
        
        # Cluster X coordinates to find columns
        columns = self._cluster_x_coordinates(x_centers, page_width)
        
        # Assign column IDs
        for region in text_regions:
            center_x = region.center[0]
            region.column_id = self._assign_to_column(center_x, columns)
        
        logging.info(f"Detected {len(columns)} columns")
        return regions
    
    def _cluster_x_coordinates(self, x_coords: List[int], page_width: int) -> List[int]:
        """Cluster X coordinates to identify column centers."""
        if not x_coords:
            return [page_width // 2]
        
        # Simple clustering: find gaps larger than min_column_gap
        sorted_x = sorted(x_coords)
        
        columns = [sorted_x[0]]
        for x in sorted_x[1:]:
            if x - columns[-1] > self.min_column_gap:
                columns.append(x)
        
        return columns
    
    def _assign_to_column(self, x: int, columns: List[int]) -> int:
        """Assign X coordinate to nearest column."""
        if not columns:
            return 0
        
        distances = [abs(x - col) for col in columns]
        return distances.index(min(distances))
    
    def _determine_reading_order(self, regions: List[LayoutRegion]) -> List[LayoutRegion]:
        """
        Determine reading order for regions.
        
        Strategy:
        1. Headers first
        2. Title
        3. Multi-column text (left-to-right, top-to-bottom per column)
        4. Footers last
        """
        # Separate by type
        headers = [r for r in regions if r.region_type == RegionType.HEADER]
        titles = [r for r in regions if r.region_type == RegionType.TITLE]
        headings = [r for r in regions if r.region_type == RegionType.HEADING]
        text_regions = [r for r in regions if r.region_type == RegionType.TEXT]
        tables = [r for r in regions if r.region_type == RegionType.TABLE]
        footers = [r for r in regions if r.region_type == RegionType.FOOTER]
        
        order = 0
        
        # Headers first (top to bottom)
        for region in sorted(headers, key=lambda r: r.bbox[1]):
            region.reading_order = order
            order += 1
        
        # Titles
        for region in sorted(titles, key=lambda r: r.bbox[1]):
            region.reading_order = order
            order += 1
        
        # Headings
        for region in sorted(headings, key=lambda r: r.bbox[1]):
            region.reading_order = order
            order += 1
        
        # Text regions by column
        if text_regions:
            # Group by column
            columns = {}
            for region in text_regions:
                col_id = region.column_id
                if col_id not in columns:
                    columns[col_id] = []
                columns[col_id].append(region)
            
            # Process each column top-to-bottom, columns left-to-right
            for col_id in sorted(columns.keys()):
                col_regions = sorted(columns[col_id], key=lambda r: r.bbox[1])
                for region in col_regions:
                    region.reading_order = order
                    order += 1
        
        # Tables
        for region in sorted(tables, key=lambda r: r.bbox[1]):
            region.reading_order = order
            order += 1
        
        # Footers last (top to bottom)
        for region in sorted(footers, key=lambda r: r.bbox[1]):
            region.reading_order = order
            order += 1
        
        return regions


class TableDetector:
    """
    Specialized detector for table structures.
    """
    
    def __init__(self):
        self.min_rows = 2
        self.min_cols = 2
    
    def detect_tables(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Detect table structures in document.
        
        Returns:
            List of bounding boxes (x1, y1, x2, y2) for detected tables
        """
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect horizontal and vertical lines
        horizontal_lines = self._detect_lines(gray, horizontal=True)
        vertical_lines = self._detect_lines(gray, horizontal=False)
        
        # Find intersections to identify table cells
        tables = self._find_table_structures(horizontal_lines, vertical_lines)
        
        return tables
    
    def _detect_lines(self, gray: np.ndarray, horizontal: bool = True) -> List:
        """Detect horizontal or vertical lines in image."""
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Create line detection kernel
        if horizontal:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        # Detect lines
        detected = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(detected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    def _find_table_structures(self, h_lines: List, v_lines: List) -> List[Tuple]:
        """Identify table structures from detected lines."""
        # Simplified: return empty for now
        # Full implementation would analyze line intersections
        return []


class ReadingOrderOptimizer:
    """
    Optimize reading order for complex layouts.
    """
    
    def optimize_reading_order(self, regions: List[LayoutRegion]) -> List[LayoutRegion]:
        """
        Refine reading order based on document structure.
        
        Handles:
        - Complex multi-column layouts
        - Mixed text and images
        - Sidebar content
        """
        # Sort by current reading order
        sorted_regions = sorted(regions, key=lambda r: r.reading_order)
        
        # Apply reading order rules
        optimized = self._apply_reading_rules(sorted_regions)
        
        # Reassign order numbers
        for idx, region in enumerate(optimized):
            region.reading_order = idx
        
        return optimized
    
    def _apply_reading_rules(self, regions: List[LayoutRegion]) -> List[LayoutRegion]:
        """Apply document-specific reading order rules."""
        # For now, return as-is
        # Future: implement smart reordering logic
        return regions


def visualize_layout(image_path: str, regions: List[LayoutRegion], output_path: str):
    """
    Create visualization of detected layout regions.
    
    Args:
        image_path: Input image path
        regions: Detected layout regions
        output_path: Where to save visualization
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    
    # Color map for region types
    colors = {
        RegionType.TEXT: (0, 255, 0),      # Green
        RegionType.TITLE: (255, 0, 0),     # Blue
        RegionType.HEADING: (0, 165, 255), # Orange
        RegionType.TABLE: (0, 255, 255),   # Yellow
        RegionType.HEADER: (255, 0, 255),  # Magenta
        RegionType.FOOTER: (128, 128, 128),# Gray
    }
    
    # Draw regions
    for region in regions:
        x1, y1, x2, y2 = region.bbox
        color = colors.get(region.region_type, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw reading order number
        cv2.putText(
            img, 
            str(region.reading_order), 
            (x1 + 5, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        
        # Draw region type
        cv2.putText(
            img,
            region.region_type.value,
            (x1 + 5, y1 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
    
    # Save visualization
    cv2.imwrite(output_path, img)
    logging.info(f"Layout visualization saved to {output_path}")

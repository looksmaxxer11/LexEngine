"""
Phase 2: Multi-Scale OCR Engine
Processes text at multiple resolutions and aggregates results for improved accuracy.
"""

from __future__ import annotations

import logging
import os
import statistics
from typing import Dict, List, Optional, Tuple
from PIL import Image
import numpy as np


class MultiScaleOCR:
    """
    Multi-scale OCR processor that runs OCR at multiple scales and selects the best result.
    
    Strategies:
    - Scale pyramid: Process at 100%, 150%, 200% scales
    - Confidence voting: Select result with highest confidence
    - Text consensus: Aggregate text from multiple scales
    """
    
    def __init__(self, ocr_engine, scales: Optional[List[float]] = None):
        """
        Initialize multi-scale OCR processor.
        
        Args:
            ocr_engine: Base OCR engine to use
            scales: List of scale factors (default: [1.0, 1.5, 2.0])
        """
        self.ocr_engine = ocr_engine
        self.scales = scales or [1.0, 1.5, 2.0]
        self.temp_dir = None
        
    def _create_scaled_image(self, image_path: str, scale: float, output_path: str) -> str:
        """Create a scaled version of the image."""
        try:
            img = Image.open(image_path)
            if scale != 1.0:
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                img = img.resize((new_width, new_height), Image.LANCZOS)
            img.save(output_path)
            return output_path
        except Exception as e:
            logging.error(f"Failed to create scaled image at {scale}x: {e}")
            return image_path
    
    def _get_confidence(self, raw_data: Optional[dict]) -> float:
        """Extract average confidence from raw OCR data."""
        if not raw_data:
            return 0.0
        try:
            rows = raw_data.get("tesseract_tsv", [])
            scores = [float(r.get("conf", -1.0)) for r in rows if float(r.get("conf", -1.0)) >= 0]
            return float(statistics.mean(scores)) if scores else 0.0
        except Exception:
            return 0.0
    
    def ocr_multiscale(
        self, 
        image_path: str, 
        lang: str = "auto",
        strategy: str = "best_confidence"
    ) -> Tuple[str, float, Optional[dict]]:
        """
        Run OCR at multiple scales and aggregate results.
        
        Args:
            image_path: Path to input image
            lang: Language code for OCR
            strategy: Aggregation strategy ('best_confidence', 'consensus', 'longest')
        
        Returns:
            Tuple of (text, confidence, raw_data)
        """
        results = []
        
        # Setup temp directory for scaled images
        if self.temp_dir is None:
            base_dir = os.path.dirname(image_path)
            self.temp_dir = os.path.join(base_dir, "temp_scaled")
            os.makedirs(self.temp_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Process at each scale
        for scale in self.scales:
            try:
                # Create scaled image
                scaled_path = os.path.join(
                    self.temp_dir, 
                    f"{base_name}_scale_{int(scale*100)}.png"
                )
                scaled_path = self._create_scaled_image(image_path, scale, scaled_path)
                
                # Run OCR
                text, raw = self.ocr_engine.ocr_region(scaled_path, lang=lang)
                conf = self._get_confidence(raw)
                
                results.append({
                    'scale': scale,
                    'text': text,
                    'confidence': conf,
                    'raw': raw,
                    'text_length': len(text.strip())
                })
                
                logging.debug(f"Scale {scale}x: conf={conf:.2f}, len={len(text)}")
                
                # Clean up scaled image
                if scale != 1.0 and os.path.exists(scaled_path):
                    try:
                        os.remove(scaled_path)
                    except:
                        pass
                        
            except Exception as e:
                logging.error(f"OCR failed at scale {scale}x: {e}")
                continue
        
        if not results:
            return "", 0.0, None
        
        # Select best result based on strategy
        if strategy == "best_confidence":
            best = max(results, key=lambda x: x['confidence'])
        elif strategy == "longest":
            best = max(results, key=lambda x: x['text_length'])
        elif strategy == "consensus":
            # Use consensus of texts weighted by confidence
            best = self._consensus_text(results)
        else:
            best = results[0]
        
        logging.info(
            f"Multi-scale OCR: selected scale {best['scale']}x "
            f"(conf={best['confidence']:.2f})"
        )
        
        return best['text'], best['confidence'], best['raw']
    
    def _consensus_text(self, results: List[dict]) -> dict:
        """Create consensus from multiple OCR results."""
        # For now, return the result with highest confidence
        # Future: Implement word-level voting
        return max(results, key=lambda x: x['confidence'])
    
    def ocr_adaptive_scale(
        self, 
        image_path: str, 
        lang: str = "auto",
        min_confidence: float = 70.0
    ) -> Tuple[str, float, Optional[dict]]:
        """
        Adaptively select scale based on initial OCR quality.
        
        Args:
            image_path: Path to input image
            lang: Language code for OCR
            min_confidence: Minimum acceptable confidence
        
        Returns:
            Tuple of (text, confidence, raw_data)
        """
        # First try at 1.0x scale
        text, conf, raw = self.ocr_engine.ocr_region(image_path, lang=lang)
        conf = self._get_confidence(raw)
        
        logging.debug(f"Initial OCR at 1.0x: conf={conf:.2f}")
        
        # If confidence is good, return immediately
        if conf >= min_confidence:
            return text, conf, raw
        
        # Try higher scales for low confidence
        logging.info(f"Low confidence ({conf:.2f}), trying higher scales...")
        return self.ocr_multiscale(image_path, lang, strategy="best_confidence")


class ScaleOptimizer:
    """
    Learns optimal scales for different document types and quality levels.
    """
    
    def __init__(self):
        self.scale_history: Dict[str, List[Tuple[float, float]]] = {}
        
    def record_result(self, doc_type: str, scale: float, confidence: float):
        """Record scale performance for learning."""
        if doc_type not in self.scale_history:
            self.scale_history[doc_type] = []
        self.scale_history[doc_type].append((scale, confidence))
    
    def get_optimal_scales(self, doc_type: str, top_k: int = 3) -> List[float]:
        """Get optimal scales for document type based on history."""
        if doc_type not in self.scale_history or not self.scale_history[doc_type]:
            return [1.0, 1.5, 2.0]  # Default scales
        
        # Calculate average confidence per scale
        scale_scores: Dict[float, List[float]] = {}
        for scale, conf in self.scale_history[doc_type]:
            if scale not in scale_scores:
                scale_scores[scale] = []
            scale_scores[scale].append(conf)
        
        # Rank scales by average confidence
        scale_avg = [(s, statistics.mean(confs)) for s, confs in scale_scores.items()]
        scale_avg.sort(key=lambda x: x[1], reverse=True)
        
        return [s for s, _ in scale_avg[:top_k]]

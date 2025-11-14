"""
Phase 2: Confidence-Based Retry System
Automatically retries low-confidence OCR with improved preprocessing and parameters.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from PIL import Image
import cv2


@dataclass
class RetryStrategy:
    """Configuration for a retry attempt."""
    name: str
    preprocessing_fn: Callable
    ocr_params: Dict
    min_trigger_conf: float  # Only trigger if confidence below this
    description: str


class ConfidenceRetryEngine:
    """
    Retry OCR with improved preprocessing when confidence is low.
    
    Retry strategies:
    1. Aggressive denoising
    2. Different binarization thresholds
    3. Morphological operations
    4. Contrast enhancement
    5. Sharpening
    """
    
    def __init__(
        self, 
        ocr_engine,
        min_confidence: float = 70.0,
        max_retries: int = 3
    ):
        """
        Initialize confidence-based retry engine.
        
        Args:
            ocr_engine: Base OCR engine to use
            min_confidence: Minimum acceptable confidence before retry
            max_retries: Maximum number of retry attempts
        """
        self.ocr_engine = ocr_engine
        self.min_confidence = min_confidence
        self.max_retries = max_retries
        self.retry_strategies = self._init_strategies()
        self.temp_dir = None
        
    def _init_strategies(self) -> List[RetryStrategy]:
        """Initialize retry strategies in order of aggressiveness."""
        return [
            RetryStrategy(
                name="aggressive_denoise",
                preprocessing_fn=self._aggressive_denoise,
                ocr_params={},
                min_trigger_conf=70.0,
                description="Aggressive noise removal with bilateral filter"
            ),
            RetryStrategy(
                name="adaptive_threshold",
                preprocessing_fn=self._adaptive_threshold,
                ocr_params={},
                min_trigger_conf=60.0,
                description="Adaptive Gaussian thresholding"
            ),
            RetryStrategy(
                name="morphological_enhance",
                preprocessing_fn=self._morphological_enhance,
                ocr_params={},
                min_trigger_conf=50.0,
                description="Morphological closing and opening"
            ),
            RetryStrategy(
                name="super_contrast",
                preprocessing_fn=self._super_contrast,
                ocr_params={},
                min_trigger_conf=40.0,
                description="Extreme contrast enhancement with CLAHE"
            ),
            RetryStrategy(
                name="sharpen",
                preprocessing_fn=self._sharpen,
                ocr_params={},
                min_trigger_conf=30.0,
                description="Sharpening filter for blurry text"
            ),
        ]
    
    def _aggressive_denoise(self, img: np.ndarray) -> np.ndarray:
        """Apply aggressive denoising."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Bilateral filter preserves edges while removing noise
        denoised = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Otsu binarization
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _adaptive_threshold(self, img: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Adaptive Gaussian thresholding
        binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def _morphological_enhance(self, img: np.ndarray) -> np.ndarray:
        """Apply morphological operations."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Otsu binarization first
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = np.ones((2, 2), np.uint8)
        
        # Closing: fills small holes
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Opening: removes small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return opened
    
    def _super_contrast(self, img: np.ndarray) -> np.ndarray:
        """Apply extreme contrast enhancement."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # CLAHE with aggressive parameters
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        
        # Otsu binarization
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _sharpen(self, img: np.ndarray) -> np.ndarray:
        """Apply sharpening filter."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Sharpening kernel
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        
        sharpened = cv2.filter2D(img, -1, kernel)
        
        # Otsu binarization
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _get_confidence(self, raw_data: Optional[dict]) -> float:
        """Extract average confidence from raw OCR data."""
        if not raw_data:
            return 0.0
        try:
            import statistics
            rows = raw_data.get("tesseract_tsv", [])
            scores = [float(r.get("conf", -1.0)) for r in rows if float(r.get("conf", -1.0)) >= 0]
            return float(statistics.mean(scores)) if scores else 0.0
        except Exception:
            return 0.0
    
    def _apply_preprocessing(self, image_path: str, preprocessing_fn: Callable) -> str:
        """Apply preprocessing and save to temp file."""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return image_path
            
            # Apply preprocessing
            processed = preprocessing_fn(img)
            
            # Save to temp file
            if self.temp_dir is None:
                base_dir = os.path.dirname(image_path)
                self.temp_dir = os.path.join(base_dir, "temp_retry")
                os.makedirs(self.temp_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            temp_path = os.path.join(self.temp_dir, f"{base_name}_preprocessed.png")
            
            cv2.imwrite(temp_path, processed)
            return temp_path
            
        except Exception as e:
            logging.error(f"Preprocessing failed: {e}")
            return image_path
    
    def ocr_with_retry(
        self, 
        image_path: str, 
        lang: str = "auto"
    ) -> Tuple[str, float, Optional[dict], int]:
        """
        Run OCR with confidence-based retry.
        
        Args:
            image_path: Path to input image
            lang: Language code for OCR
        
        Returns:
            Tuple of (text, confidence, raw_data, retry_count)
        """
        # Initial attempt
        text, raw = self.ocr_engine.ocr_region(image_path, lang=lang)
        conf = self._get_confidence(raw)
        
        logging.info(f"Initial OCR: conf={conf:.2f}")
        
        # If confidence is acceptable, return
        if conf >= self.min_confidence:
            return text, conf, raw, 0
        
        # Try retry strategies
        best_result = (text, conf, raw)
        retry_count = 0
        
        for strategy in self.retry_strategies:
            # Only try if confidence is below trigger threshold
            if conf >= strategy.min_trigger_conf:
                continue
            
            if retry_count >= self.max_retries:
                break
            
            retry_count += 1
            logging.info(
                f"Retry {retry_count}/{self.max_retries}: {strategy.name} "
                f"(current conf={conf:.2f})"
            )
            
            try:
                # Apply preprocessing
                processed_path = self._apply_preprocessing(
                    image_path, 
                    strategy.preprocessing_fn
                )
                
                # Run OCR on preprocessed image
                retry_text, retry_raw = self.ocr_engine.ocr_region(
                    processed_path, 
                    lang=lang
                )
                retry_conf = self._get_confidence(retry_raw)
                
                logging.info(f"  {strategy.name} result: conf={retry_conf:.2f}")
                
                # Keep if better
                if retry_conf > conf:
                    conf = retry_conf
                    best_result = (retry_text, retry_conf, retry_raw)
                    logging.info(f"  ✅ Improved confidence to {retry_conf:.2f}")
                
                # Clean up temp file
                if processed_path != image_path and os.path.exists(processed_path):
                    try:
                        os.remove(processed_path)
                    except:
                        pass
                
                # Stop if we reached acceptable confidence
                if conf >= self.min_confidence:
                    logging.info(f"✅ Reached acceptable confidence: {conf:.2f}")
                    break
                    
            except Exception as e:
                logging.error(f"Retry strategy {strategy.name} failed: {e}")
                continue
        
        final_text, final_conf, final_raw = best_result
        
        if retry_count > 0:
            logging.info(
                f"Retry summary: {retry_count} attempts, "
                f"final conf={final_conf:.2f}"
            )
        
        return final_text, final_conf, final_raw, retry_count


class SmartRetryOrchestrator:
    """
    Orchestrates multi-scale OCR and confidence-based retry.
    """
    
    def __init__(
        self,
        ocr_engine,
        min_confidence: float = 70.0,
        enable_multiscale: bool = True,
        enable_retry: bool = True
    ):
        """
        Initialize smart retry orchestrator.
        
        Args:
            ocr_engine: Base OCR engine
            min_confidence: Minimum acceptable confidence
            enable_multiscale: Enable multi-scale OCR
            enable_retry: Enable confidence-based retry
        """
        self.ocr_engine = ocr_engine
        self.min_confidence = min_confidence
        self.enable_multiscale = enable_multiscale
        self.enable_retry = enable_retry
        
        self.multiscale = None
        self.retry_engine = None
        
        if enable_multiscale:
            from .multiscale_ocr import MultiScaleOCR
            self.multiscale = MultiScaleOCR(ocr_engine)
        
        if enable_retry:
            self.retry_engine = ConfidenceRetryEngine(ocr_engine, min_confidence)
    
    def ocr_smart(
        self, 
        image_path: str, 
        lang: str = "auto"
    ) -> Tuple[str, float, Optional[dict]]:
        """
        Run OCR with smart retry strategies.
        
        Workflow:
        1. Try standard OCR
        2. If low confidence and multiscale enabled, try multi-scale
        3. If still low confidence and retry enabled, try preprocessing retry
        
        Args:
            image_path: Path to input image
            lang: Language code for OCR
        
        Returns:
            Tuple of (text, confidence, raw_data)
        """
        # Initial attempt
        text, raw = self.ocr_engine.ocr_region(image_path, lang=lang)
        conf = self._get_confidence(raw)
        
        logging.info(f"Smart OCR initial: conf={conf:.2f}")
        
        if conf >= self.min_confidence:
            return text, conf, raw
        
        # Try multi-scale if enabled
        if self.enable_multiscale and self.multiscale:
            logging.info("Trying multi-scale OCR...")
            ms_text, ms_conf, ms_raw = self.multiscale.ocr_multiscale(
                image_path, lang, strategy="best_confidence"
            )
            if ms_conf > conf:
                text, conf, raw = ms_text, ms_conf, ms_raw
                logging.info(f"Multi-scale improved confidence to {conf:.2f}")
        
        if conf >= self.min_confidence:
            return text, conf, raw
        
        # Try confidence retry if enabled
        if self.enable_retry and self.retry_engine:
            logging.info("Trying confidence-based retry...")
            retry_text, retry_conf, retry_raw, _ = self.retry_engine.ocr_with_retry(
                image_path, lang
            )
            if retry_conf > conf:
                text, conf, raw = retry_text, retry_conf, retry_raw
                logging.info(f"Retry improved confidence to {conf:.2f}")
        
        return text, conf, raw
    
    def _get_confidence(self, raw_data: Optional[dict]) -> float:
        """Extract average confidence from raw OCR data."""
        if not raw_data:
            return 0.0
        try:
            import statistics
            rows = raw_data.get("tesseract_tsv", [])
            scores = [float(r.get("conf", -1.0)) for r in rows if float(r.get("conf", -1.0)) >= 0]
            return float(statistics.mean(scores)) if scores else 0.0
        except Exception:
            return 0.0

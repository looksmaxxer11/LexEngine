"""
Qwen-VL OCR Engine - AI-based vision-language model for document OCR
Supports multilingual text including Cyrillic scripts (Russian, Uzbek)
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class QwenOCREngine:
    """
    Qwen-VL based OCR engine with multilingual support.
    Handles Russian, Uzbek (Latin/Cyrillic), English and mixed scripts.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen-VL-Chat", device: Optional[str] = None):
        """
        Initialize Qwen-VL model for OCR.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load Qwen-VL model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading Qwen-VL model: {self.model_name} on {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model with appropriate settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device if self.device == 'cuda' else 'cpu',
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            ).eval()
            
            logger.info("Qwen-VL model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen-VL model: {e}")
            raise
    
    def extract_text(self, image_path: str, lang: Optional[str] = None) -> Tuple[str, float]:
        """
        Extract text from image using Qwen-VL.
        
        Args:
            image_path: Path to image file
            lang: Language hint (ignored - Qwen-VL is multilingual)
        
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Qwen-VL model not loaded")
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Create OCR prompt for Qwen-VL
            query = self.tokenizer.from_list_format([
                {'image': image_path},
                {'text': 'Read all text from this document image. Extract the text exactly as it appears, preserving the layout and order.'},
            ])
            
            # Generate text extraction
            with torch.no_grad():
                response, _ = self.model.chat(
                    self.tokenizer,
                    query=query,
                    history=None
                )
            
            # Qwen-VL doesn't provide confidence scores directly
            # We estimate confidence based on response length and quality
            confidence = min(0.95, 0.7 + (len(response) / 1000) * 0.2)
            
            logger.info(f"Qwen-VL extracted {len(response)} characters from {image_path}")
            
            return response.strip(), confidence
            
        except Exception as e:
            logger.error(f"Qwen-VL OCR failed for {image_path}: {e}")
            return "", 0.0
    
    def extract_text_with_details(self, image_path: str, lang: Optional[str] = None) -> dict:
        """
        Extract text with detailed information.
        
        Args:
            image_path: Path to image file
            lang: Language hint (optional)
        
        Returns:
            Dictionary with text, confidence, and metadata
        """
        text, confidence = self.extract_text(image_path, lang)
        
        return {
            'text': text,
            'confidence': confidence,
            'engine': 'qwen-vl',
            'model': self.model_name,
            'device': self.device,
            'multilingual': True,
            'supports_cyrillic': True
        }


def create_qwen_engine(model_name: str = "Qwen/Qwen-VL-Chat", device: Optional[str] = None) -> QwenOCREngine:
    """
    Factory function to create Qwen OCR engine.
    
    Args:
        model_name: Hugging Face model identifier
        device: Device to run on
    
    Returns:
        Initialized QwenOCREngine instance
    """
    return QwenOCREngine(model_name=model_name, device=device)

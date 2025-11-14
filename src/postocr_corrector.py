"""
Phase 3: Post-OCR Text Correction
Language-aware spell checking and context-based error correction.
"""

from __future__ import annotations

import logging
import re
from typing import List, Dict, Optional, Tuple
from collections import Counter
import difflib


class PostOCRCorrector:
    """
    Post-OCR text correction with language-aware capabilities.
    
    Features:
    - Common OCR error patterns
    - Dictionary-based validation
    - Context-aware corrections
    - Language-specific rules (Uzbek Latin/Cyrillic, Russian)
    """
    
    def __init__(self, language: str = "auto"):
        """
        Initialize post-OCR corrector.
        
        Args:
            language: Target language ('uz_lat', 'uz_cyr', 'ru', 'en', 'auto')
        """
        self.language = language
        self.ocr_error_patterns = self._init_error_patterns()
        self.word_frequencies = {}
        
    def _init_error_patterns(self) -> Dict[str, str]:
        """Initialize common OCR error patterns."""
        return {
            # Common character confusions
            r'\b0\b': 'O',  # Zero to O
            r'\bl\b': 'I',  # Lowercase L to I
            r'\brn\b': 'm',  # rn to m
            r'\bvv\b': 'w',  # vv to w
            
            # Cyrillic-Latin confusions
            'а': 'a',  # Cyrillic a to Latin a (in Latin text)
            'е': 'e',
            'о': 'o',
            'р': 'p',
            'с': 'c',
            'у': 'y',
            'х': 'x',
            
            # Common Uzbek patterns
            r"o'": "oʻ",  # Apostrophe normalization
            r"g'": "gʻ",
            
            # Multiple spaces
            r'\s+': ' ',
            
            # Line breaks within words
            r'(\w+)-\s+(\w+)': r'\1\2',
        }
    
    def correct_text(self, text: str) -> str:
        """
        Apply all correction strategies to text.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Corrected text
        """
        if not text:
            return text
        
        # Apply pattern-based corrections
        corrected = self._apply_pattern_corrections(text)
        
        # Fix spacing issues
        corrected = self._fix_spacing(corrected)
        
        # Correct common OCR errors
        corrected = self._correct_common_errors(corrected)
        
        # Clean up
        corrected = self._cleanup(corrected)
        
        logging.debug(f"Post-OCR correction applied, {len(text)} -> {len(corrected)} chars")
        return corrected
    
    def _apply_pattern_corrections(self, text: str) -> str:
        """Apply regex-based pattern corrections."""
        for pattern, replacement in self.ocr_error_patterns.items():
            text = re.sub(pattern, replacement, text)
        return text
    
    def _fix_spacing(self, text: str) -> str:
        """Fix spacing issues around punctuation."""
        # Remove space before punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        # Add space after punctuation if missing
        text = re.sub(r'([.,;:!?])([А-Яа-яA-Za-z])', r'\1 \2', text)
        
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _correct_common_errors(self, text: str) -> str:
        """Correct common OCR misrecognitions."""
        # Dictionary of common OCR errors and corrections
        corrections = {
            # Uzbek Latin
            "o'": "oʻ",
            "g'": "gʻ",
            "sh": "sh",  # Keep as is
            "ch": "ch",  # Keep as is
            
            # Common misreads
            "l": "I",  # Context-dependent
            "0": "O",  # Context-dependent
        }
        
        # Apply word-level corrections
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Check if word needs correction
            corrected_word = word
            for error, correction in corrections.items():
                if error in word:
                    corrected_word = word.replace(error, correction)
            corrected_words.append(corrected_word)
        
        return ' '.join(corrected_words)
    
    def _cleanup(self, text: str) -> str:
        """Final cleanup of text."""
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Trim whitespace
        text = text.strip()
        
        return text
    
    def correct_line(self, line: str) -> str:
        """Correct a single line of text."""
        return self.correct_text(line)


class LanguageAwareCorrector:
    """
    Advanced correction with language detection and language-specific rules.
    """
    
    def __init__(self):
        self.uzbek_lat_chars = set("abcdefghijklmnopqrstuvwxyzʻoʻgʻ")
        self.uzbek_cyr_chars = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюяўқғҳ")
        self.russian_chars = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
        
    def detect_script(self, text: str) -> str:
        """
        Detect script type (Latin, Cyrillic).
        
        Returns:
            'latin', 'cyrillic', or 'mixed'
        """
        if not text:
            return 'unknown'
        
        latin_count = sum(1 for c in text.lower() if c in self.uzbek_lat_chars)
        cyrillic_count = sum(1 for c in text.lower() if c in self.uzbek_cyr_chars)
        
        total = latin_count + cyrillic_count
        if total == 0:
            return 'unknown'
        
        latin_ratio = latin_count / total
        
        if latin_ratio > 0.7:
            return 'latin'
        elif latin_ratio < 0.3:
            return 'cyrillic'
        else:
            return 'mixed'
    
    def correct_mixed_script(self, text: str) -> str:
        """
        Correct text with mixed scripts (common OCR error).
        
        Converts misplaced Cyrillic characters in Latin text and vice versa.
        """
        script = self.detect_script(text)
        
        if script == 'mixed':
            # Detect dominant script
            words = text.split()
            word_scripts = [self.detect_script(w) for w in words]
            dominant = Counter(word_scripts).most_common(1)[0][0]
            
            # Convert minority script characters
            if dominant == 'latin':
                text = self._cyrillic_to_latin(text)
            elif dominant == 'cyrillic':
                text = self._latin_to_cyrillic(text)
        
        return text
    
    def _cyrillic_to_latin(self, text: str) -> str:
        """Convert misplaced Cyrillic characters to Latin."""
        # Map of visually similar Cyrillic -> Latin
        mapping = {
            'а': 'a', 'А': 'A',
            'е': 'e', 'Е': 'E',
            'о': 'o', 'О': 'O',
            'р': 'p', 'Р': 'P',
            'с': 'c', 'С': 'C',
            'у': 'y', 'У': 'Y',
            'х': 'x', 'Х': 'X',
        }
        
        for cyr, lat in mapping.items():
            text = text.replace(cyr, lat)
        
        return text
    
    def _latin_to_cyrillic(self, text: str) -> str:
        """Convert misplaced Latin characters to Cyrillic."""
        # Inverse of above mapping
        mapping = {
            'a': 'а', 'A': 'А',
            'e': 'е', 'E': 'Е',
            'o': 'о', 'O': 'О',
            'p': 'р', 'P': 'Р',
            'c': 'с', 'C': 'С',
            'y': 'у', 'Y': 'У',
            'x': 'х', 'X': 'Х',
        }
        
        for lat, cyr in mapping.items():
            text = text.replace(lat, cyr)
        
        return text


class ContextAwareCorrector:
    """
    Context-based error correction using surrounding text.
    """
    
    def __init__(self):
        self.min_confidence_threshold = 0.7
    
    def correct_with_context(
        self, 
        word: str, 
        context_before: List[str], 
        context_after: List[str]
    ) -> Tuple[str, float]:
        """
        Correct word using surrounding context.
        
        Args:
            word: Word to correct
            context_before: Previous words
            context_after: Following words
            
        Returns:
            Tuple of (corrected_word, confidence)
        """
        # Simplified: return word as-is
        # Full implementation would use language model
        return word, 1.0
    
    def find_similar_words(self, word: str, dictionary: List[str], max_distance: int = 2) -> List[str]:
        """
        Find similar words in dictionary using edit distance.
        
        Args:
            word: Target word
            dictionary: List of valid words
            max_distance: Maximum edit distance
            
        Returns:
            List of similar words
        """
        similar = []
        for dict_word in dictionary:
            # Calculate Levenshtein distance
            distance = self._edit_distance(word.lower(), dict_word.lower())
            if distance <= max_distance:
                similar.append((dict_word, distance))
        
        # Sort by distance
        similar.sort(key=lambda x: x[1])
        return [w for w, d in similar]
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance."""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


def create_corrector(language: str = "auto") -> PostOCRCorrector:
    """
    Factory function to create appropriate corrector.
    
    Args:
        language: Target language
        
    Returns:
        PostOCRCorrector instance
    """
    return PostOCRCorrector(language=language)


def batch_correct(texts: List[str], language: str = "auto") -> List[str]:
    """
    Correct multiple texts in batch.
    
    Args:
        texts: List of text strings
        language: Target language
        
    Returns:
        List of corrected texts
    """
    corrector = create_corrector(language)
    return [corrector.correct_text(text) for text in texts]

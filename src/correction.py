from __future__ import annotations

from typing import Optional
import re


class TextCorrector:
    """
    OCR post-correction using a fine-tuned seq2seq model if available,
    otherwise applies heuristic character-level fixes.
    """

    def __init__(self, model_dir: Optional[str] = None, device: Optional[str] = None) -> None:
        self.model = None
        self.tokenizer = None
        self.device = device or "cpu"

        if model_dir:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
                try:
                    import torch  # type: ignore
                    self.model.to(self.device)
                except Exception:
                    pass
            except Exception:
                # If transformers or model not available, fallback will be used
                self.model = None
                self.tokenizer = None

    def correct(self, text: str, max_new_tokens: int = 2048) -> str:
        if not text:
            return text
        if self.model and self.tokenizer:
            try:
                import torch  # type: ignore
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                return self.tokenizer.decode(output[0], skip_special_tokens=True)
            except Exception:
                pass
        # Heuristic fixes for common OCR confusions across Cyrillic/Latin
        cleaned = text
        subs = {
            "\u041e": "O",  # Cyrillic O → Latin O
            "\u043e": "o",  # Cyrillic o → Latin o
            "\u0406": "I",  # Cyrillic I → Latin I
            "\u0456": "i",  # Cyrillic i → Latin i
            r"0(?!\d)": "O",  # zero not in number → O
            r"(?<=\D)1(?=\D)": "l",  # 1 between non-digits → l
        }
        for pat, rep in subs.items():
            cleaned = re.sub(pat, rep, cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned


__all__ = ["TextCorrector"]

from __future__ import annotations

from typing import Dict, Optional, Tuple, List
import os
import statistics


class OCREngine:
    """
    Region OCR powered by Tesseract (via pytesseract).

    Language routing:
    - uz_cyr → rus (Cyrillic)
    - uz_lat → eng (Latin)
    - ru → rus
    - en → eng
    - multi/auto → eng
    """

    def __init__(self, use_angle_cls: bool = True, gpu: bool = False, det_db_thresh: float = 0.3, psm: int = 1) -> None:
        # Keep signature for compatibility; options are ignored for Tesseract.
        # psm: Page Segmentation Mode (1=auto with OSD, 3=fully auto, 11=sparse text)
        self._engines: Dict[str, bool] = {}
        self.psm = psm
        self._configure_tesseract_cmd()

    def _configure_tesseract_cmd(self) -> None:
        try:
            import pytesseract
            # Allow override via env var
            tcmd = os.environ.get("TESSERACT_CMD")
            if tcmd and os.path.isfile(tcmd):
                pytesseract.pytesseract.tesseract_cmd = tcmd
                return
            # Common Windows install path
            default_path = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            if os.path.isfile(default_path):
                pytesseract.pytesseract.tesseract_cmd = default_path
        except Exception:
            pass

    @staticmethod
    def _map_lang(lang: str) -> str:
        """Map language codes to Tesseract language packs.

        Notes:
        - 'uzb' is the Uzbek traineddata in tessdata.
        - Combining languages (e.g., 'uzb+eng+rus') often improves accuracy for mixed announcements.
        - Dynamically inspects available languages from current tessdata so we never request a missing pack.
        """
        l = (lang or "").lower().strip()

        # Cache available tessdata languages in function attribute
        try:
            import pytesseract
            if not hasattr(OCREngine._map_lang, "_avail"):
                try:
                    OCREngine._map_lang._avail = set(pytesseract.get_languages(config=""))
                except Exception:
                    OCREngine._map_lang._avail = set()
            avail: set[str] = getattr(OCREngine._map_lang, "_avail", set())
        except Exception:
            avail = set()

        # Helper to build a + joined string only with actually available packs
        def combo(order: List[str]) -> str:
            present = [p for p in order if p in avail]
            # If none matched (e.g., enumeration failed) fall back to order as-is
            return "+".join(present or order)

        base_all = ["uzb", "eng", "rus"]  # desired order (Uzbek first improves its script bias)

        if l in {"uz_cyr", "uz-cyr", "uz-cyrillic", "cyrillic"}:
            # Cyrillic heavy: prioritize rus + uzb; include eng if present for stray Latin words
            return combo(["rus", "uzb", "eng"])
        if l in {"uz_lat", "uz-lat", "uz-latin", "latin", "uz-latin"}:
            # Latin Uzbek plus possible Cyrillic tokens
            return combo(["uzb", "eng", "rus"])
        if l in {"ru", "rus", "russian"}:
            return combo(["rus", "uzb", "eng"])
        if l in {"en", "eng", "english"}:
            return combo(["eng", "uzb", "rus"])
        if l in {"multi", "auto", ""}:
            return combo(base_all)
        # Default fallback tries all to maximize recall.
        return combo(base_all)

    def _image_to_data(self, image_path: str, lang: str) -> List[dict]:
        import pytesseract
        from PIL import Image
        import cv2
        import numpy as np

        # Load and preprocess image aggressively for newspaper quality
        try:
            img_pil = Image.open(image_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open image for OCR: {e}")
        
        # Convert to grayscale numpy array
        img_np = np.array(img_pil.convert('L'))
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_np = clahe.apply(img_np)
        
        # Otsu binarization (auto threshold)
        _, img_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Noise removal (morphological closing)
        kernel = np.ones((2, 2), np.uint8)
        img_np = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to PIL
        img = Image.fromarray(img_np)
        
        # Use instance PSM setting
        config = os.environ.get("TESSERACT_CONFIG", f"--oem 3 --psm {self.psm}")

        def run_with_lang(l: str):
            return pytesseract.image_to_data(img, lang=l, config=config, output_type=pytesseract.Output.DICT)

        try:
            tsv = run_with_lang(lang)
        except Exception as e:
            # Fallback if a language (e.g., 'uzb') is missing
            lang_parts = [p for p in lang.split('+') if p]
            if 'uzb' in lang_parts:
                try_lang = '+'.join([p for p in lang_parts if p != 'uzb']) or 'eng'
                try:
                    tsv = run_with_lang(try_lang)
                except Exception:
                    tsv = run_with_lang('eng')
            else:
                # Last resort to English
                tsv = run_with_lang('eng')
        rows: List[dict] = []
        n = len(tsv.get("text", []))
        for i in range(n):
            text = (tsv["text"][i] or "").strip()
            conf = tsv.get("conf", ["-1"][0]) if isinstance(tsv.get("conf"), list) else "-1"
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = -1.0
            if text:
                rows.append({
                    "text": text,
                    "conf": conf_f,
                    "left": int(tsv.get("left", [0]*n)[i] or 0),
                    "top": int(tsv.get("top", [0]*n)[i] or 0),
                    "width": int(tsv.get("width", [0]*n)[i] or 0),
                    "height": int(tsv.get("height", [0]*n)[i] or 0),
                })
        return rows

    def ocr_region(self, image_path: str, lang: str = "auto") -> Tuple[str, Optional[dict]]:
        mapped = self._map_lang(lang)
        rows = self._image_to_data(image_path, mapped)
        lines = [r["text"] for r in rows]
        return "\n".join(lines).strip(), {"tesseract_tsv": rows}


def _avg_confidence(raw: Optional[dict]) -> float:
        if not raw:
            return 0.0
        try:
            rows = raw.get("tesseract_tsv") or []
            scores = [float(r.get("conf", -1.0)) for r in rows if float(r.get("conf", -1.0)) >= 0]
            return float(statistics.mean(scores)) if scores else 0.0
        except Exception:
            return 0.0


def ocr_region_with_conf(engine: OCREngine, image_path: str, lang: str = "auto") -> Tuple[str, float, Optional[dict]]:
    text, raw = engine.ocr_region(image_path, lang=lang)
    return text, _avg_confidence(raw), raw


def detect_language_fast(text: str) -> Optional[str]:
    """fastText language detection if available; returns canonical tag or None."""
    try:
        import importlib
        spec = importlib.util.find_spec("fasttext")
        if spec is None:
            return None
        fasttext = importlib.import_module("fasttext")

        model_path = os.environ.get("FASTTEXT_LID_MODEL", "lid.176.bin")
        if not hasattr(detect_language_fast, "_model"):
            detect_language_fast._model = fasttext.load_model(model_path)
        labels, _ = detect_language_fast._model.predict(text.replace("\n", " ")[:4000])
        label = labels[0] if labels else ""
        if "__label__ru" in label:
            return "ru"
        if "__label__en" in label:
            return "en"
        if "__label__uz" in label:
            return "uz-cyrillic" if any("\u0400" <= ch <= "\u04FF" for ch in text) else "uz-latin"
        return None
    except Exception:
        return None


__all__ = ["OCREngine", "ocr_region_with_conf", "detect_language_fast"]

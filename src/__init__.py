"""
Fresh OCR → AI Structuring Pipeline (greenfield implementation).

Modules:
- preprocessing: PDF→images, OpenCV normalization
- layout: region detection (PP-Structure optional, heuristic fallback)
- ocr: PaddleOCR-backed region OCR (multi-language routing)
- correction: OCR post-correction (ByT5/T5 LoRA optional, heuristic fallback)
- structuring: regex + heuristics to schema
- embeddings: sentence-transformers embeddings
- storage: Qdrant/Postgres adapters
- schema: Pydantic/dataclass models for stable JSON output

All modules aim to be import-safe on Windows; heavy optional deps
are imported lazily and guarded with helpful error messages.
"""

__all__ = [
    "preprocessing",
    "layout",
    "ocr",
    "correction",
    "structuring",
    "embeddings",
    "storage",
    "schema",
]

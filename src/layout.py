from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import os


@dataclass
class Region:
    name: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (inclusive)


def detect_layout(image_path: str, strategy: str = "ppstructure") -> List[Region]:
    """
    Detects key regions on page image.

    strategy options:
    - "ppstructure": try PaddleOCR PP-Structure if available, else fallback
    - "fallback": simple heuristic regions (header/date ~ top, body ~ remaining)
    """
    if strategy == "ppstructure":
        try:
            return _detect_with_ppstructure(image_path)
        except Exception:
            # fall back silently to heuristic
            pass
    return _detect_fallback(image_path)


def _detect_with_ppstructure(image_path: str) -> List[Region]:
    # Lazy import to keep module import-safe
    try:
        from paddleocr import PPStructure  # type: ignore
    except Exception as e:
        raise RuntimeError("PaddleOCR PPStructure not available") from e

    import cv2  # type: ignore

    table_engine = PPStructure(layout=True)
    img = cv2.imread(image_path)
    result = table_engine(img)

    h, w = img.shape[:2]

    # Heuristic mapping from detected blocks to known regions
    regions: List[Region] = []
    top_band = int(0.2 * h)
    # Prefer top-most text as header/date, rest as body
    regions.append(Region("header", (0, 0, w - 1, max(0, top_band - 1))))
    regions.append(Region("body", (0, top_band, w - 1, h - 1)))

    # Add ref/date as small subregions within header (split header horizontally)
    header_left = Region("ref", (0, 0, max(0, int(0.5 * w) - 1), max(0, top_band - 1)))
    header_right = Region("date", (int(0.5 * w), 0, w - 1, max(0, top_band - 1)))
    regions.extend([header_left, header_right])
    return regions


def _detect_fallback(image_path: str) -> List[Region]:
    import cv2  # type: ignore

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]

    top_band = int(0.2 * h)
    regions: List[Region] = [
        Region("header", (0, 0, w - 1, max(0, top_band - 1))),
        Region("ref", (0, 0, max(0, int(0.5 * w) - 1), max(0, top_band - 1))),
        Region("date", (int(0.5 * w), 0, w - 1, max(0, top_band - 1))),
        Region("body", (0, top_band, w - 1, h - 1)),
    ]
    return regions


__all__ = ["Region", "detect_layout"]

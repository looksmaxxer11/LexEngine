from __future__ import annotations

import re
from typing import Dict, Optional


REF_PAT = re.compile(r"(?:№|No\.?|N\.?|#)\s*([0-9]+\/[0-9]{4})", re.IGNORECASE)
DATE_PAT = re.compile(r"(\d{1,2}[\.\-/]\d{1,2}[\.\-/]\d{4})")


def detect_script_language(text: str) -> str:
    """Heuristic language/script detection. Returns one of 'uz-latin', 'uz-cyrillic', 'ru', 'en'."""
    if not text:
        return "en"
    # Cyrillic block detection
    if re.search(r"[\u0400-\u04FF]", text):
        # Distinguish Russian vs Uzbek Cyrillic is non-trivial; default to uz-cyrillic
        return "uz-cyrillic"
    # Latin: could be en or uz-latin; default to uz-latin for this corpus
    return "uz-latin"


def extract_ref(text: str) -> Optional[str]:
    m = REF_PAT.search(text or "")
    if m:
        return m.group(1)
    return None


def extract_date(text: str) -> Optional[str]:
    m = DATE_PAT.search(text or "")
    if m:
        # Normalize to YYYY-MM-DD if possible
        raw = m.group(1)
        parts = re.split(r"[\./-]", raw)
        if len(parts) == 3:
            d, mth, y = parts
            try:
                dd = int(d)
                mm = int(mth)
                yyyy = int(y)
                return f"{yyyy:04d}-{mm:02d}-{dd:02d}"
            except Exception:
                return raw
        return raw
    return None


__all__ = [
    "detect_script_language",
    "extract_ref",
    "extract_date",
    "split_announcements",
]


UPPER_LINE = re.compile(r"^[A-Z0-9\-\s]{20,}$")
REF_TOKEN = re.compile(r"(?:№|No\.?|N\.?)\s*\d+")
DATE_TOKEN = re.compile(r"\b\d{1,2}[\./-]\d{1,2}[\./-]\d{4}\b")


def _is_boundary_line(line: str) -> bool:
    """Heuristic to decide if a line likely starts a new announcement.

    Conditions (any true):
    - Contains a date token near start
    - Contains a reference token (№123/2022 etc.)
    - Is long uppercase-ish banner line (often separating blocks)
    - Very short separator like just 'OOO' or repeated pattern tokens
    """
    s = line.strip()
    if not s:
        return False
    if DATE_TOKEN.search(s[:30]):
        return True
    if REF_TOKEN.search(s):
        return True
    # Uppercase ratio
    letters = [c for c in s if c.isalpha()]
    if letters:
        upper_ratio = sum(c.isupper() for c in letters) / max(len(letters), 1)
        if upper_ratio > 0.85 and len(s) > 25:
            return True
    # Simple token separators
    if s in {"OOO", "==="}:
        return True
    if UPPER_LINE.match(s):
        return True
    return False


def split_announcements(text: str) -> list[str]:
    """Split a large OCR/corrected text blob into announcement segments.

    Strategy:
    - Scan lines; when a boundary line encountered and current segment has content -> start new.
    - Merge consecutive boundary lines into a single segment header.
    - Filter out leading/trailing empty lines per segment.
    - Discard segments with < 40 alphabetic chars (likely noise) unless it's the only segment.
    """
    if not text:
        return []
    lines = text.splitlines()
    segments: list[list[str]] = []
    current: list[str] = []

    def flush():
        if current:
            segments.append(current.copy())
            current.clear()

    for line in lines:
        if _is_boundary_line(line) and current:
            flush()
            current.append(line)  # start new with boundary line
        else:
            current.append(line)
    flush()

    cleaned: list[str] = []
    for seg in segments:
        # Trim empties
        while seg and not seg[0].strip():
            seg.pop(0)
        while seg and not seg[-1].strip():
            seg.pop()
        joined = "\n".join(seg).strip()
        alpha = sum(c.isalpha() for c in joined)
        if alpha < 40 and len(segments) > 1:
            continue
        cleaned.append(joined)
    return cleaned

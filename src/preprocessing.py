from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def preprocess_pdf(
    input_pdf: str,
    out_dir: str,
    dpi: int = 300,
) -> List[str]:
    """
    Convert a PDF into preprocessed PNG pages suitable for OCR.

    Steps:
    - pdf2image (requires Poppler on Windows)
    - grayscale, denoise, binarize (OTSU), light deskew attempt

    Returns a list of absolute file paths for preprocessed page images.
    """
    ensure_dir(out_dir)

    # Try pdf2image (Poppler). If not available, fall back to PyMuPDF.
    convert_from_path = None
    pdfinfo_error = None
    try:
        from pdf2image import convert_from_path as _cfp
        convert_from_path = _cfp
    except Exception:
        convert_from_path = None

    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenCV (cv2) is required for preprocessing.") from e

    pages = None
    if convert_from_path is not None:
        try:
            pages = convert_from_path(input_pdf, dpi=dpi)
        except Exception as e:
            # pdf2image raises PDFInfoNotInstalledError when Poppler is missing on Windows
            pdfinfo_error = e

    if pages is None:
        # Fallback: render via PyMuPDF (MuPDF) without external binaries
        pages = _render_with_pymupdf(input_pdf, dpi)
    output_paths: List[str] = []

    for i, page in enumerate(pages):
        # page can be a PIL.Image (from pdf2image) or a numpy array (from PyMuPDF fallback)
        img = np.array(page) if hasattr(page, 'size') else page
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 3)
        _thr, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # very light deskew (estimate via Hough lines if any)
        try:
            binary = _deskew(binary)
        except Exception:
            pass

        out_path = os.path.join(out_dir, f"page_{i:04d}.png")
        cv2.imwrite(out_path, binary)
        output_paths.append(os.path.abspath(out_path))

    return output_paths


def _deskew(binary: np.ndarray) -> np.ndarray:
    import cv2  # type: ignore

    coords = np.column_stack(np.where(binary == 0))  # text pixels
    if coords.size == 0:
        return binary
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _render_with_pymupdf(input_pdf: str, dpi: int) -> List[np.ndarray]:
    """
    Fallback renderer using PyMuPDF (fitz) to avoid Poppler dependency on Windows.
    Returns a list of numpy arrays (BGR) for each page image.
    """
    try:
        import fitz  # PyMuPDF
        import cv2  # type: ignore
        import numpy as np  # local alias
    except Exception as e:
        raise RuntimeError(
            "Neither Poppler+pdf2image nor PyMuPDF are available. Install one of: \n"
            "  pip install pymupdf  (recommended on Windows)\n"
            "  or install Poppler and keep pdf2image"
        ) from e

    doc = fitz.open(input_pdf)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    images: List[np.ndarray] = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        # Decode PNG bytes to numpy BGR array
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        images.append(img)
    doc.close()
    return images


__all__ = ["preprocess_pdf", "ensure_dir"]

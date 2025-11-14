from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional, Tuple

from .preprocessing import preprocess_pdf, ensure_dir
from .advanced_preprocessing import process_pdf_with_line_detection, adaptive_script_detection
from .layout import detect_layout, Region
from .ocr import OCREngine, ocr_region_with_conf, detect_language_fast
from .correction import TextCorrector
from .structuring import extract_ref, extract_date, detect_script_language, split_announcements
from .embeddings import EmbeddingGenerator
from .schema import AnnouncementRecord
from .storage import QdrantStore, PostgresStore
from .postprocess import (
    clean_text,
    extract_tables_as_text,
    deduplicate_lines,
    filter_low_confidence_lines,
    normalize_text_lines,
)


def _setup_logging(data_root: str) -> None:
    ensure_dir(os.path.join(data_root, "logs"))
    log_path = os.path.join(data_root, "logs", "pipeline.log")
    logging.basicConfig(
        filename=log_path,
        filemode="a",
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )
    logging.getLogger().addHandler(logging.StreamHandler())


def _ocr_page(
    page_img: str,
    layout_strategy: str,
    ocr_engine: Optional[OCREngine],
    data_root: str,
) -> str:
    if ocr_engine is None:
        return "[DRY-RUN] OCR text placeholder for pipeline wiring."
    import cv2  # type: ignore
    regions = detect_layout(page_img, strategy=layout_strategy)
    body_regions = [r for r in regions if r.name == "body"] or regions

    texts: List[str] = []
    for r in body_regions:
        img = cv2.imread(page_img)
        x1, y1, x2, y2 = r.bbox
        crop = img[y1 : y2 + 1, x1 : x2 + 1]
        crop_dir = os.path.join(data_root, "images")
        ensure_dir(crop_dir)
        crop_path = os.path.join(crop_dir, f"{os.path.basename(page_img)}_{r.name}.png")
        cv2.imwrite(crop_path, crop)

        # Ensemble: try latin and ru, pick higher confidence
        text_lat, conf_lat, _ = ocr_region_with_conf(ocr_engine, crop_path, lang="latin")
        text_cyr, conf_cyr, _ = ocr_region_with_conf(ocr_engine, crop_path, lang="ru")
        final_text = text_lat if conf_lat >= conf_cyr - 0.05 else text_cyr
        texts.append(final_text)
    return "\n".join(texts)


def run_pipeline(
    input_pdf: str,
    output_json: str,
    data_root: str = "data",
    model_dir: Optional[str] = None,
    layout_strategy: str = "ppstructure",
    use_gpu: bool = False,
    enable_embeddings: bool = True,
    store: str = "none",
    dry_run: bool = False,
    max_workers: int = None,
    use_advanced_preprocessing: bool = True,
    detect_separator_lines: bool = True,
    zoom_scale: float = 1.5,
    enable_noise_reduction: bool = True,
    ocr_engine: str = "tesseract",
) -> AnnouncementRecord:
    data_root = os.path.abspath(data_root)
    ensure_dir(data_root)
    ensure_dir(os.path.join(data_root, "images"))
    ensure_dir(os.path.join(data_root, "json"))
    ensure_dir(os.path.join(data_root, "embeddings"))
    _setup_logging(data_root)

    # Auto-detect optimal worker count if not specified
    if max_workers is None:
        import multiprocessing as mp
        max_workers = max(1, mp.cpu_count() - 1)  # Use all cores except 1
        logging.info(f"Auto-detected {mp.cpu_count()} CPU cores, using {max_workers} workers")

    input_pdf = os.path.abspath(input_pdf)
    output_json = os.path.abspath(output_json)
    ensure_dir(os.path.dirname(output_json))

    logging.info("Starting pipeline for %s", input_pdf)

    # 1) Try direct PDF text extraction (bypass OCR if reliable)
    direct_text_pages = [] if dry_run else _try_pdf_text_extraction(input_pdf)
    reliable_direct_text = _is_reliable_text(direct_text_pages)

    # 2) Preprocess images only if OCR is needed (300 DPI for newspaper quality)
    page_images = []
    if not reliable_direct_text:
        if use_advanced_preprocessing:
            # Use advanced preprocessing with line detection and zooming
            page_images = [] if dry_run else process_pdf_with_line_detection(
                input_pdf,
                os.path.join(data_root, "images"),
                dpi=300,
                zoom_scale=zoom_scale,
                detect_lines=detect_separator_lines
            )
        else:
            # Use legacy preprocessing
            page_images = [] if dry_run else preprocess_pdf(
                input_pdf,
                os.path.join(data_root, "images"),
                dpi=300
            )
        if dry_run:
            page_images = [input_pdf]

    # 2+3) Layout + OCR (parallel per page) or use direct PDF text
    raw_texts: List[str] = []
    if reliable_direct_text and direct_text_pages:
        raw_texts = direct_text_pages
        logging.info("Using direct PDF text extraction for %d pages (no OCR)", len(raw_texts))
    else:
        # Initialize OCR engine based on selection
        ocr = None
        if not dry_run:
            if ocr_engine == "qwen":
                from .qwen_ocr import create_qwen_engine
                logging.info("Initializing Qwen-VL OCR engine...")
                qwen_model = create_qwen_engine(device='cuda' if use_gpu else 'cpu')
                # Wrapper to match Tesseract interface
                class QwenWrapper:
                    def __init__(self, qwen):
                        self.qwen = qwen
                    def image_to_text(self, img_path, lang=None):
                        text, _ = self.qwen.extract_text(img_path, lang)
                        return text
                ocr = QwenWrapper(qwen_model)
                logging.info("Qwen-VL OCR engine initialized")
            else:  # Default to tesseract
                # PSM 1 for multi-column newspaper layouts with OSD
                ocr = OCREngine(use_angle_cls=True, gpu=use_gpu, psm=1)
                logging.info("Tesseract OCR engine initialized")
        
        # Process pages in parallel with progress tracking
        from concurrent.futures import as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            # Submit all tasks
            futures = {ex.submit(_ocr_page, p, layout_strategy, ocr, data_root): i 
                      for i, p in enumerate(page_images)}
            
            # Collect results as they complete (maintains order)
            results_dict = {}
            completed = 0
            total = len(page_images)
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results_dict[idx] = result
                    completed += 1
                    logging.info(f"OCR progress: {completed}/{total} pages completed ({100*completed//total}%)")
                except Exception as e:
                    logging.error(f"OCR failed for page {idx}: {e}")
                    results_dict[idx] = ""
            
            # Reconstruct ordered results
            results = [results_dict[i] for i in range(len(page_images))]
            raw_texts.extend(results)

    # Stage texts
    raw_original = "\n\n".join(t for t in raw_texts if t)
    
    # Apply noise reduction if enabled
    if enable_noise_reduction and raw_original:
        logging.info("Applying noise reduction filters...")
        # Deduplicate repeated lines (watermarks, headers)
        raw_original = deduplicate_lines(raw_original, threshold=3)
        # Normalize text
        raw_original = normalize_text_lines(raw_original)
        # Filter low-confidence heuristically
        raw_original = filter_low_confidence_lines(raw_original)

    # Post-process: clean whitespace, fix spacing, extract tables
    clean_stage = clean_text(raw_original)
    clean_stage = extract_tables_as_text(clean_stage)

    # 4) Correction
    corrector = TextCorrector(model_dir=model_dir)
    corrected = corrector.correct(clean_stage)

    # 5) Structuring
    ref = extract_ref(corrected)
    date = extract_date(corrected)
    # Language detection: fastText if available, else heuristic
    language_pred = detect_language_fast(corrected) or detect_script_language(corrected)

    # 5b) Announcement segmentation (raw / clean / corrected)
    segments_raw = split_announcements(raw_original)
    segments_clean = split_announcements(clean_stage)
    segments_corrected = split_announcements(corrected)

    record = AnnouncementRecord(
        ref=ref,
        date=date,
        language=language_pred,
        announcement=corrected,
        source_pdf=input_pdf,
        metadata={
            "processed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "pages": len(page_images),
            "raw_text": raw_original,
            "clean_text": clean_stage,
            "segments_raw": segments_raw,
            "segments_clean": segments_clean,
            "segments_corrected": segments_corrected,
        },
    )

    # 6) Embeddings
    if enable_embeddings and not dry_run:
        try:
            eg = EmbeddingGenerator()
            record.embedding = eg.encode(record.announcement or "")
        except Exception as e:
            logging.warning("Embeddings skipped: %s", e)
            record.metadata["embeddings_error"] = str(e)

    # Store
    if store in {"qdrant", "postgres"} and not dry_run:
        try:
            if store == "qdrant" and record.embedding:
                q = QdrantStore()
                q.upsert(
                    ids=[f"{os.path.basename(input_pdf)}:{ref or 'n/a'}"],
                    vectors=[record.embedding],
                    payloads=[{
                        "ref": record.ref,
                        "date": record.date,
                        "language": record.language,
                        "source_pdf": record.source_pdf,
                    }],
                )
            elif store == "postgres":
                dsn = os.environ.get("PG_DSN", "dbname=postgres user=postgres password=postgres host=localhost port=5432")
                p = PostgresStore(dsn)
                p.upsert_record({
                    "ref": record.ref,
                    "date": record.date,
                    "language": record.language,
                    "announcement": record.announcement,
                    "source_pdf": record.source_pdf,
                    "metadata": record.metadata,
                })
        except Exception as e:
            logging.error("Storage error: %s", e)
            record.metadata["storage_error"] = str(e)

    # Write JSON
    with open(output_json, "w", encoding="utf-8") as f:
        f.write(record.to_json())
    logging.info("Wrote %s", output_json)
    return record


def _try_pdf_text_extraction(input_pdf: str) -> List[str]:
    """Extract per-page text using PyMuPDF. Returns list of page texts."""
    try:
        import fitz  # type: ignore
    except Exception:
        return []
    texts: List[str] = []
    try:
        doc = fitz.open(input_pdf)
        for page in doc:
            t = page.get_text("text") or ""
            # Normalize ligatures and weird spaces
            t = t.replace("\u00A0", " ")
            texts.append(t.strip())
        doc.close()
    except Exception:
        return []
    return texts


def _is_reliable_text(pages: List[str]) -> bool:
    """Heuristic: consider direct text reliable if enough alphabetic content.

    - At least one page has >= 200 alphabetic characters
    - Combined alphabetic ratio >= 0.3
    """
    if not pages:
        return False
    total = "\n\n".join(pages)
    alpha = sum(ch.isalpha() for ch in total)
    length = max(len(total), 1)
    has_long_page = any(sum(ch.isalpha() for ch in p) >= 200 for p in pages)
    return has_long_page and (alpha / length) >= 0.30


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Legal Announcement OCR → AI Structuring Pipeline")
    p.add_argument("--input", required=True, help="Input PDF path")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--data-root", default="data", help="Root folder for all processed data")
    p.add_argument("--model", default=None, help="Path to fine-tuned corrector model directory")
    p.add_argument("--layout-strategy", default="ppstructure", choices=["ppstructure", "fallback"], help="Layout detection backend")
    p.add_argument("--gpu", action="store_true", help="Use GPU if supported by Paddle")
    p.add_argument("--no-embeddings", action="store_true", help="Disable embeddings stage")
    p.add_argument("--store", choices=["none", "qdrant", "postgres"], default="none", help="Where to store results")
    p.add_argument("--dry-run", action="store_true", help="Run without heavy deps; produce placeholder OCR output")
    p.add_argument("--max-workers", type=int, default=None, help="Parallel workers for OCR (None=auto-detect all cores-1)")
    p.add_argument("--use-advanced-preprocessing", action="store_true", default=True, help="Use enhanced preprocessing with line detection")
    p.add_argument("--no-line-detection", action="store_true", help="Disable separator line detection")
    p.add_argument("--zoom-scale", type=float, default=1.5, help="Zoom scale for image enhancement (default: 1.5)")
    p.add_argument("--no-noise-reduction", action="store_true", help="Disable noise reduction filters")
    p.add_argument("--ocr-engine", choices=["tesseract", "qwen"], default="tesseract", help="OCR engine to use (tesseract or qwen)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_pipeline(
        input_pdf=args.input,
        output_json=args.output,
        data_root=args.data_root,
        model_dir=args.model,
        layout_strategy=args.layout_strategy,
        use_gpu=args.gpu,
        enable_embeddings=not args.no_embeddings,
        store=args.store,
        dry_run=args.dry_run,
        max_workers=args.max_workers,
        use_advanced_preprocessing=args.use_advanced_preprocessing,
        detect_separator_lines=not args.no_line_detection,
        zoom_scale=args.zoom_scale,
        enable_noise_reduction=not args.no_noise_reduction,
        ocr_engine=args.ocr_engine,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

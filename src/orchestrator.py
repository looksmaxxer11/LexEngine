from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional

from .preprocessing import preprocess_pdf, ensure_dir
from .layout import detect_layout, Region
from .ocr import OCREngine, ocr_region_with_conf, detect_language_fast
from .correction import TextCorrector
from .structuring import extract_ref, extract_date, detect_script_language
from .embeddings import EmbeddingGenerator
from .schema import AnnouncementRecord
from .storage import QdrantStore, PostgresStore
from .confidence_retry import SmartRetryOrchestrator
from .layout_analyzer import LayoutAnalyzer, visualize_layout
from .postocr_corrector import PostOCRCorrector, LanguageAwareCorrector
from .region_ocr import RegionOCR


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
    use_phase2: bool = False,
    use_region_ocr: bool = False,
) -> str:
    if ocr_engine is None:
        return "[DRY-RUN] OCR text placeholder for pipeline wiring."
    import cv2  # type: ignore
    img = cv2.imread(page_img)
    if img is None:
        return ""
    h, w = img.shape[:2]

    # Phase 3.5: Use RegionOCR for proper column ordering (TOP-TO-BOTTOM reading)
    if use_region_ocr:
        region_ocr = RegionOCR(ocr_engine, min_column_gap=50, max_workers=8)  # 8 parallel OCR threads
        text, regions = region_ocr.process_image_with_reading_order(img)
        logging.info(f"✅ RegionOCR: Processed {len(regions)} regions in CORRECT READING ORDER (top-to-bottom)")
        return text

    regions = detect_layout(page_img, strategy=layout_strategy)
    body_regions = [r for r in regions if r.name == "body"] or regions

    # Initialize Phase 2 smart retry if enabled
    smart_ocr = None
    if use_phase2:
        smart_ocr = SmartRetryOrchestrator(
            ocr_engine, 
            min_confidence=70.0,
            enable_multiscale=True,
            enable_retry=True
        )

    texts: List[str] = []
    for idx, r in enumerate(body_regions):
        x1, y1, x2, y2 = r.bbox
        # Clamp to image bounds
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1 : y2 + 1, x1 : x2 + 1]
        if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            continue

        crop_dir = os.path.join(data_root, "images")
        ensure_dir(crop_dir)
        crop_path = os.path.join(crop_dir, f"{os.path.basename(page_img)}_{r.name}_{idx}.png")
        cv2.imwrite(crop_path, crop)

        if use_phase2 and smart_ocr:
            # Use Phase 2 smart OCR with multi-scale and retry
            final_text, final_conf, _ = smart_ocr.ocr_smart(crop_path, lang="auto")
            if final_text.strip():
                texts.append(final_text)
        else:
            # Ensemble: try English (Latin) and Russian (Cyrillic), pick higher confidence
            text_lat, conf_lat, _ = ocr_region_with_conf(ocr_engine, crop_path, lang="en")
            text_cyr, conf_cyr, _ = ocr_region_with_conf(ocr_engine, crop_path, lang="ru")
            final_text = text_lat if conf_lat >= conf_cyr - 0.05 else text_cyr
            if final_text.strip():
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
    max_workers: int = 2,
    quick: bool = False,
    ocr_engine: Optional[OCREngine] = None,
    use_phase2: bool = False,
    use_phase3: bool = False,
    use_region_ocr: bool = True,  # Phase 3.5: Enable by default for correct reading order
) -> AnnouncementRecord:
    data_root = os.path.abspath(data_root)
    ensure_dir(data_root)
    ensure_dir(os.path.join(data_root, "images"))
    ensure_dir(os.path.join(data_root, "json"))
    ensure_dir(os.path.join(data_root, "embeddings"))
    _setup_logging(data_root)

    input_pdf = os.path.abspath(input_pdf)
    output_json = os.path.abspath(output_json)
    ensure_dir(os.path.dirname(output_json))

    logging.info("Starting pipeline for %s", input_pdf)

    # 1) Preprocess - quality boost in quick mode for Tesseract
    dpi = 300 if quick else 300
    page_images = [] if dry_run else preprocess_pdf(input_pdf, os.path.join(data_root, "images"), dpi=dpi)
    if dry_run:
        page_images = [input_pdf]

    # 2+3) OCR - Quick mode or Advanced mode
    if quick:
        # QUICK MODE: Fast full-page OCR, no layout detection
        logging.info("✅ Quick mode: Full-page OCR (fast path)")
        if ocr_engine is None and not dry_run:
            logging.info("Creating OCR engine...")
            ocr_engine = OCREngine(use_angle_cls=False, gpu=use_gpu)
        
        # Initialize Phase 2 smart retry if enabled
        smart_ocr = None
        if use_phase2 and not dry_run:
            logging.info("🚀 Phase 2 optimizations enabled: Multi-scale + Confidence Retry")
            smart_ocr = SmartRetryOrchestrator(
                ocr_engine, 
                min_confidence=70.0,
                enable_multiscale=True,
                enable_retry=True
            )
        
        raw_texts: List[str] = []
        if not dry_run:
            total_pages = len(page_images)
            logging.info(f"📄 Processing {total_pages} pages with Quick OCR...")
            for idx, page_img in enumerate(page_images, 1):
                try:
                    logging.info(f"  Page {idx}/{total_pages} - Processing...")
                    
                    if use_phase2 and smart_ocr:
                        # Use Phase 2 smart OCR
                        text, conf, _ = smart_ocr.ocr_smart(page_img, lang="auto")
                        logging.info(f"  Page {idx}/{total_pages} - ✅ Complete (conf={conf:.2f}, {int(idx/total_pages*100)}%)")
                    else:
                        # Standard OCR
                        text, _ = ocr_engine.ocr_region(page_img, lang="auto")
                        logging.info(f"  Page {idx}/{total_pages} - ✅ Complete ({int(idx/total_pages*100)}%)")
                    
                    if text.strip():
                        raw_texts.append(text)
                except Exception as e:
                    logging.error(f"  Page {idx}/{total_pages} - ❌ Error: {e}")
                    raw_texts.append(f"[ERROR processing page {idx}: {e}]")
        else:
            raw_texts = ["[DRY-RUN] Quick OCR placeholder"]
        
        combined_raw = "\n\n".join(raw_texts)
        
        # Apply Phase 3 post-OCR correction if enabled
        if use_phase3:
            logging.info("🔧 Phase 3: Applying post-OCR correction...")
            post_corrector = PostOCRCorrector(language="auto")
            lang_corrector = LanguageAwareCorrector()
            corrected = post_corrector.correct_text(combined_raw)
            corrected = lang_corrector.correct_mixed_script(corrected)
            logging.info("✅ Phase 3 post-OCR correction complete!")
        else:
            corrected = combined_raw  # Skip correction in quick mode
        
        logging.info("✅ Quick OCR complete!")
    else:
        # ADVANCED MODE: Layout detection + region OCR
        logging.info("⚠️ Advanced mode: Layout detection + region OCR")
        if use_phase2:
            logging.info("🚀 Phase 2 optimizations enabled for advanced mode")
        logging.info("   This will load PP-Structure models (30-60s first time)")
        if ocr_engine is None and not dry_run:
            ocr_engine = OCREngine(use_angle_cls=True, gpu=use_gpu)
        
        total_pages = len(page_images)
        logging.info(f"📄 Processing {total_pages} pages with layout detection...")
        raw_texts: List[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(
                ex.map(lambda p: _ocr_page(p, layout_strategy, ocr_engine, data_root, use_phase2, use_region_ocr), page_images)
            )
            raw_texts.extend(results)
        combined_raw = "\n\n".join(t for t in raw_texts if t)
        
        # 4) Correction
        logging.info("🔧 Running text correction...")
        corrector = TextCorrector(model_dir=model_dir)
        corrected = corrector.correct(combined_raw)
        
        # Apply Phase 3 post-OCR correction if enabled
        if use_phase3:
            logging.info("🔧 Phase 3: Applying post-OCR correction...")
            post_corrector = PostOCRCorrector(language="auto")
            lang_corrector = LanguageAwareCorrector()
            corrected = post_corrector.correct_text(corrected)
            corrected = lang_corrector.correct_mixed_script(corrected)
            logging.info("✅ Phase 3 post-OCR correction complete!")
        
        logging.info("✅ Advanced OCR complete!")

    # 5) Structuring
    ref = extract_ref(corrected)
    date = extract_date(corrected)
    # Language detection: fastText if available, else heuristic
    language_pred = detect_language_fast(corrected) or detect_script_language(corrected)

    record = AnnouncementRecord(
        ref=ref,
        date=date,
        language=language_pred,
        announcement=corrected,
        source_pdf=input_pdf,
        metadata={
            "processed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "pages": len(page_images),
        },
    )

    # 6) Embeddings
    if enable_embeddings and not dry_run and not quick:
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
    p.add_argument("--max-workers", type=int, default=2, help="Parallel workers for OCR")
    p.add_argument("--phase2", action="store_true", help="Enable Phase 2 optimizations (multi-scale + confidence retry)")
    p.add_argument("--phase3", action="store_true", help="Enable Phase 3 optimizations (layout analysis + post-OCR correction)")
    p.add_argument("--no-region-ocr", action="store_true", help="Disable Phase 3.5 reading order correction (column top-to-bottom, enabled by default)")
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
        use_phase2=args.phase2,
        use_phase3=args.phase3,
        use_region_ocr=not args.no_region_ocr,  # Phase 3.5 enabled by default
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

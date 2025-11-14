"""
Phase 2 Optimizations Benchmark
Tests multi-scale OCR and confidence-based retry improvements
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ocr import OCREngine
from src.multiscale_ocr import MultiScaleOCR
from src.confidence_retry import ConfidenceRetryEngine, SmartRetryOrchestrator
from PIL import Image
import statistics


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


def create_test_images(pdf_path: str, output_dir: str):
    """Convert PDF to test images."""
    import fitz  # PyMuPDF
    
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    
    images = []
    for page_num in range(min(3, len(doc))):  # Test first 3 pages
        page = doc[page_num]
        pix = page.get_pixmap(dpi=300)
        img_path = os.path.join(output_dir, f"test_page_{page_num}.png")
        pix.save(img_path)
        images.append(img_path)
    
    doc.close()
    return images


def benchmark_standard_ocr(engine, image_path):
    """Benchmark standard OCR."""
    start = time.time()
    text, raw = engine.ocr_region(image_path, lang="auto")
    elapsed = time.time() - start
    
    # Calculate confidence
    try:
        rows = raw.get("tesseract_tsv", [])
        scores = [float(r.get("conf", -1.0)) for r in rows if float(r.get("conf", -1.0)) >= 0]
        conf = statistics.mean(scores) if scores else 0.0
    except:
        conf = 0.0
    
    return {
        'method': 'Standard OCR',
        'time': elapsed,
        'confidence': conf,
        'text_length': len(text),
        'text': text
    }


def benchmark_multiscale_ocr(engine, image_path):
    """Benchmark multi-scale OCR."""
    multiscale = MultiScaleOCR(engine, scales=[1.0, 1.5, 2.0])
    
    start = time.time()
    text, conf, raw = multiscale.ocr_multiscale(
        image_path, 
        lang="auto", 
        strategy="best_confidence"
    )
    elapsed = time.time() - start
    
    return {
        'method': 'Multi-Scale OCR',
        'time': elapsed,
        'confidence': conf,
        'text_length': len(text),
        'text': text
    }


def benchmark_confidence_retry(engine, image_path):
    """Benchmark confidence-based retry."""
    retry_engine = ConfidenceRetryEngine(
        engine, 
        min_confidence=70.0, 
        max_retries=3
    )
    
    start = time.time()
    text, conf, raw, retry_count = retry_engine.ocr_with_retry(image_path, lang="auto")
    elapsed = time.time() - start
    
    return {
        'method': f'Confidence Retry ({retry_count} retries)',
        'time': elapsed,
        'confidence': conf,
        'text_length': len(text),
        'text': text,
        'retry_count': retry_count
    }


def benchmark_smart_orchestrator(engine, image_path):
    """Benchmark smart orchestrator (combines all)."""
    smart = SmartRetryOrchestrator(
        engine,
        min_confidence=70.0,
        enable_multiscale=True,
        enable_retry=True
    )
    
    start = time.time()
    text, conf, raw = smart.ocr_smart(image_path, lang="auto")
    elapsed = time.time() - start
    
    return {
        'method': 'Smart Orchestrator (Phase 2)',
        'time': elapsed,
        'confidence': conf,
        'text_length': len(text),
        'text': text
    }


def print_results_table(results_by_image):
    """Print benchmark results in table format."""
    print("\n" + "="*100)
    print("PHASE 2 OPTIMIZATION BENCHMARK RESULTS")
    print("="*100 + "\n")
    
    for img_name, results in results_by_image.items():
        print(f"\n📄 {img_name}")
        print("-" * 100)
        print(f"{'Method':<35} {'Time (s)':<12} {'Confidence':<15} {'Text Length':<15} {'Notes':<20}")
        print("-" * 100)
        
        baseline = None
        for result in results:
            method = result['method']
            time_str = f"{result['time']:.3f}"
            conf_str = f"{result['confidence']:.2f}"
            length_str = str(result['text_length'])
            
            # Calculate speedup/improvement
            notes = ""
            if method == "Standard OCR":
                baseline = result
            elif baseline:
                time_ratio = baseline['time'] / result['time'] if result['time'] > 0 else 0
                conf_diff = result['confidence'] - baseline['confidence']
                notes = f"Δconf: {conf_diff:+.2f}"
                if 'retry_count' in result and result['retry_count'] > 0:
                    notes += f" ({result['retry_count']} retries)"
            
            print(f"{method:<35} {time_str:<12} {conf_str:<15} {length_str:<15} {notes:<20}")
        
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_phase2_optimizations.py <path_to_pdf>")
        print("Example: python test_phase2_optimizations.py data/raw_pdfs/sample.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    print("\n" + "="*100)
    print("PHASE 2 OPTIMIZATION BENCHMARK")
    print("="*100)
    print(f"\nInput PDF: {pdf_path}")
    print("\nPhase 2 Features:")
    print("  ✅ Multi-scale OCR (1.0x, 1.5x, 2.0x)")
    print("  ✅ Confidence-based retry (up to 3 attempts)")
    print("  ✅ Smart orchestrator (adaptive strategy)")
    print("\n" + "-"*100)
    
    # Setup
    temp_dir = "temp_benchmark"
    logging.info("Converting PDF to test images...")
    test_images = create_test_images(pdf_path, temp_dir)
    logging.info(f"Created {len(test_images)} test images")
    
    # Initialize OCR engine
    logging.info("Initializing OCR engine...")
    engine = OCREngine(use_angle_cls=False, gpu=False, psm=3)
    
    # Run benchmarks
    results_by_image = {}
    
    for img_path in test_images:
        img_name = os.path.basename(img_path)
        logging.info(f"\n{'='*50}")
        logging.info(f"Benchmarking: {img_name}")
        logging.info(f"{'='*50}")
        
        results = []
        
        # 1. Standard OCR
        logging.info("\n1️⃣ Running Standard OCR...")
        try:
            result = benchmark_standard_ocr(engine, img_path)
            results.append(result)
            logging.info(f"   Time: {result['time']:.3f}s, Conf: {result['confidence']:.2f}")
        except Exception as e:
            logging.error(f"   Failed: {e}")
        
        # 2. Multi-scale OCR
        logging.info("\n2️⃣ Running Multi-Scale OCR...")
        try:
            result = benchmark_multiscale_ocr(engine, img_path)
            results.append(result)
            logging.info(f"   Time: {result['time']:.3f}s, Conf: {result['confidence']:.2f}")
        except Exception as e:
            logging.error(f"   Failed: {e}")
        
        # 3. Confidence Retry
        logging.info("\n3️⃣ Running Confidence Retry...")
        try:
            result = benchmark_confidence_retry(engine, img_path)
            results.append(result)
            logging.info(f"   Time: {result['time']:.3f}s, Conf: {result['confidence']:.2f}")
        except Exception as e:
            logging.error(f"   Failed: {e}")
        
        # 4. Smart Orchestrator
        logging.info("\n4️⃣ Running Smart Orchestrator (Phase 2)...")
        try:
            result = benchmark_smart_orchestrator(engine, img_path)
            results.append(result)
            logging.info(f"   Time: {result['time']:.3f}s, Conf: {result['confidence']:.2f}")
        except Exception as e:
            logging.error(f"   Failed: {e}")
        
        results_by_image[img_name] = results
    
    # Print results
    print_results_table(results_by_image)
    
    # Summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100 + "\n")
    
    all_methods = {}
    for results in results_by_image.values():
        for result in results:
            method = result['method'].split('(')[0].strip()  # Normalize method name
            if method not in all_methods:
                all_methods[method] = {'times': [], 'confs': []}
            all_methods[method]['times'].append(result['time'])
            all_methods[method]['confs'].append(result['confidence'])
    
    print(f"{'Method':<35} {'Avg Time (s)':<15} {'Avg Confidence':<15}")
    print("-" * 65)
    
    for method, data in all_methods.items():
        avg_time = statistics.mean(data['times'])
        avg_conf = statistics.mean(data['confs'])
        print(f"{method:<35} {avg_time:<15.3f} {avg_conf:<15.2f}")
    
    print("\n" + "="*100)
    print("✅ PHASE 2 BENCHMARK COMPLETE")
    print("="*100 + "\n")
    
    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    print("Note: Phase 2 optimizations are most effective on low-quality or challenging documents.")
    print("For high-quality documents, standard OCR may be sufficient.\n")


if __name__ == "__main__":
    main()

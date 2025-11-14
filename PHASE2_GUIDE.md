# Phase 2 Optimizations - Documentation

## Overview

Phase 2 optimizations significantly improve OCR accuracy and robustness through **multi-scale processing** and **confidence-based retry mechanisms**. These features are particularly effective for challenging documents with poor quality, noise, or varied text sizes.

## Features

### 1. Multi-Scale OCR
Processes text at multiple resolutions to capture details that might be missed at a single scale.

**How it works:**
- Processes images at 1.0x, 1.5x, and 2.0x scales
- Runs OCR on each scaled version
- Selects the result with highest confidence
- Particularly effective for small text or varied font sizes

**Strategies:**
- `best_confidence`: Select result with highest OCR confidence
- `longest`: Select result with most text extracted
- `consensus`: Weight results by confidence (future: word-level voting)

### 2. Confidence-Based Retry
Automatically retries OCR with improved preprocessing when confidence is low.

**How it works:**
- Initial OCR attempt
- If confidence < threshold (default 70%), trigger retry
- Apply progressively aggressive preprocessing strategies
- Keep best result (up to 3 retries)

**Retry Strategies (in order):**
1. **Aggressive Denoise** (trigger: conf < 70%)
   - Bilateral filter to remove noise while preserving edges
   - Otsu binarization for optimal thresholding

2. **Adaptive Threshold** (trigger: conf < 60%)
   - Adaptive Gaussian thresholding
   - Handles varying lighting/background

3. **Morphological Enhancement** (trigger: conf < 50%)
   - Closing operation to fill small holes
   - Opening operation to remove noise

4. **Super Contrast** (trigger: conf < 40%)
   - Aggressive CLAHE for extreme contrast enhancement
   - Best for faded or low-contrast documents

5. **Sharpening** (trigger: conf < 30%)
   - Sharpening kernel for blurry text
   - Last resort for extremely degraded documents

### 3. Smart Orchestrator
Combines multi-scale and retry strategies adaptively.

**Workflow:**
1. Try standard OCR
2. If low confidence → Try multi-scale OCR
3. If still low confidence → Try confidence-based retry
4. Return best result

## Usage

### Command Line

```bash
# Run pipeline with Phase 2 optimizations
python -m src.orchestrator --input document.pdf --output result.json --phase2

# Using the quick-start script
phase2_quickstart.bat "path/to/document.pdf"
```

### Programmatic Usage

```python
from src.ocr import OCREngine
from src.confidence_retry import SmartRetryOrchestrator

# Initialize
engine = OCREngine()
smart_ocr = SmartRetryOrchestrator(
    engine,
    min_confidence=70.0,
    enable_multiscale=True,
    enable_retry=True
)

# Run smart OCR
text, confidence, raw_data = smart_ocr.ocr_smart("image.png", lang="auto")
```

### Individual Components

```python
# Multi-scale only
from src.multiscale_ocr import MultiScaleOCR

multiscale = MultiScaleOCR(engine, scales=[1.0, 1.5, 2.0])
text, conf, raw = multiscale.ocr_multiscale(
    "image.png", 
    lang="auto", 
    strategy="best_confidence"
)

# Confidence retry only
from src.confidence_retry import ConfidenceRetryEngine

retry_engine = ConfidenceRetryEngine(engine, min_confidence=70.0, max_retries=3)
text, conf, raw, retry_count = retry_engine.ocr_with_retry("image.png", lang="auto")
```

## Benchmarking

Run the Phase 2 benchmark to compare performance:

```bash
# Using batch script
run_phase2_benchmark.bat "path/to/document.pdf"

# Or directly
python test_phase2_optimizations.py "path/to/document.pdf"
```

**Benchmark Output:**
- Processing time for each method
- OCR confidence scores
- Text extraction quality
- Comparative analysis

## Performance Characteristics

### When Phase 2 Helps Most

✅ **High Impact:**
- Low-quality scans
- Faded or degraded documents
- Varied text sizes (small + large)
- Poor lighting or contrast
- Noisy backgrounds
- Blurry images

❌ **Low Impact:**
- High-quality scans
- Clean, well-contrasted documents
- Uniform text sizes

### Performance Considerations

**Time vs Quality Trade-off:**
- Standard OCR: Fastest (baseline)
- Multi-scale: ~3x slower (3 scales)
- Confidence retry: Variable (0-3 retries)
- Smart orchestrator: Adaptive (skips strategies when confidence is good)

**Best Practices:**
1. Start with standard OCR
2. Enable Phase 2 for problem documents
3. Use `min_confidence` parameter to control aggressiveness
4. Monitor retry counts to identify problem areas

## Configuration

### Parameters

```python
# Multi-scale OCR
scales = [1.0, 1.5, 2.0]  # Scale factors to try

# Confidence retry
min_confidence = 70.0      # Trigger threshold
max_retries = 3            # Maximum retry attempts

# Smart orchestrator
enable_multiscale = True   # Enable multi-scale
enable_retry = True        # Enable retry
```

### Tuning Tips

**For Speed:**
- Use fewer scales: `[1.0, 1.5]`
- Increase `min_confidence` threshold (e.g., 80)
- Reduce `max_retries` (e.g., 2)

**For Quality:**
- Use more scales: `[1.0, 1.25, 1.5, 1.75, 2.0]`
- Lower `min_confidence` threshold (e.g., 60)
- Increase `max_retries` (e.g., 5)

## Integration with Pipeline

Phase 2 optimizations integrate seamlessly with the existing pipeline:

```bash
# Quick mode + Phase 2
python -m src.orchestrator --input doc.pdf --output result.json --quick --phase2

# Advanced mode + Phase 2 (with layout detection)
python -m src.orchestrator --input doc.pdf --output result.json --phase2

# With other options
python -m src.orchestrator \
  --input doc.pdf \
  --output result.json \
  --phase2 \
  --max-workers 4 \
  --gpu
```

## Troubleshooting

### High Processing Time
- **Cause:** All scales/retries being triggered
- **Solution:** Increase `min_confidence` threshold or reduce scales

### Low Confidence Despite Retries
- **Cause:** Document quality too poor
- **Solution:** Consider manual preprocessing or document re-scanning

### Memory Issues
- **Cause:** Large images at 2.0x scale
- **Solution:** Reduce maximum scale to 1.5x or process pages individually

## Logging

Phase 2 provides detailed logging:

```
INFO Smart OCR initial: conf=65.23
INFO Trying multi-scale OCR...
INFO Multi-scale OCR: selected scale 1.5x (conf=72.45)
INFO Multi-scale improved confidence to 72.45
```

Enable debug logging for more details:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Planned improvements:
- Word-level consensus voting
- Learned scale optimization per document type
- GPU-accelerated multi-scale processing
- Parallel retry strategies
- Custom preprocessing strategy definitions

## API Reference

### SmartRetryOrchestrator

```python
class SmartRetryOrchestrator:
    def __init__(
        self,
        ocr_engine,
        min_confidence: float = 70.0,
        enable_multiscale: bool = True,
        enable_retry: bool = True
    )
    
    def ocr_smart(
        self, 
        image_path: str, 
        lang: str = "auto"
    ) -> Tuple[str, float, Optional[dict]]
```

### MultiScaleOCR

```python
class MultiScaleOCR:
    def __init__(
        self, 
        ocr_engine, 
        scales: Optional[List[float]] = None
    )
    
    def ocr_multiscale(
        self, 
        image_path: str, 
        lang: str = "auto",
        strategy: str = "best_confidence"
    ) -> Tuple[str, float, Optional[dict]]
```

### ConfidenceRetryEngine

```python
class ConfidenceRetryEngine:
    def __init__(
        self, 
        ocr_engine,
        min_confidence: float = 70.0,
        max_retries: int = 3
    )
    
    def ocr_with_retry(
        self, 
        image_path: str, 
        lang: str = "auto"
    ) -> Tuple[str, float, Optional[dict], int]
```

## Examples

### Example 1: Quick Scan with Phase 2

```bash
phase2_quickstart.bat "C:\scans\announcement.pdf"
```

### Example 2: Batch Processing with Phase 2

```python
from src.ocr import OCREngine
from src.confidence_retry import SmartRetryOrchestrator
from pathlib import Path

engine = OCREngine()
smart_ocr = SmartRetryOrchestrator(engine)

for pdf_path in Path("scans").glob("*.pdf"):
    # Process each page
    text, conf, _ = smart_ocr.ocr_smart(str(pdf_path), lang="auto")
    print(f"{pdf_path.name}: conf={conf:.2f}")
```

### Example 3: Custom Configuration

```python
# Aggressive quality mode
smart_ocr = SmartRetryOrchestrator(
    engine,
    min_confidence=60.0,  # Lower threshold
    enable_multiscale=True,
    enable_retry=True
)

# Speed mode
smart_ocr = SmartRetryOrchestrator(
    engine,
    min_confidence=80.0,  # Higher threshold
    enable_multiscale=False,  # Skip multi-scale
    enable_retry=True
)
```

## Conclusion

Phase 2 optimizations provide substantial improvements in OCR accuracy for challenging documents, with adaptive strategies that balance speed and quality. Use the benchmark tools to validate improvements on your specific document types.

For support or questions, see the main README.md or contact the development team.

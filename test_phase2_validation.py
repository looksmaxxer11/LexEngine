"""
Quick Phase 2 Validation Test
Tests that Phase 2 modules are properly integrated and functional
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("="*60)
print("Phase 2 Quick Validation Test")
print("="*60)

# Test 1: Import modules
print("\n✅ Test 1: Importing Phase 2 modules...")
try:
    from src.multiscale_ocr import MultiScaleOCR
    from src.confidence_retry import ConfidenceRetryEngine, SmartRetryOrchestrator
    print("   ✓ All Phase 2 modules imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize OCR engine
print("\n✅ Test 2: Initializing OCR engine...")
try:
    from src.ocr import OCREngine
    engine = OCREngine(use_angle_cls=False, gpu=False)
    print("   ✓ OCR engine initialized")
except Exception as e:
    print(f"   ✗ Initialization failed: {e}")
    sys.exit(1)

# Test 3: Create MultiScaleOCR
print("\n✅ Test 3: Creating MultiScaleOCR instance...")
try:
    multiscale = MultiScaleOCR(engine, scales=[1.0, 1.5])
    print("   ✓ MultiScaleOCR created")
    print(f"   Scales: {multiscale.scales}")
except Exception as e:
    print(f"   ✗ Creation failed: {e}")
    sys.exit(1)

# Test 4: Create ConfidenceRetryEngine
print("\n✅ Test 4: Creating ConfidenceRetryEngine...")
try:
    retry_engine = ConfidenceRetryEngine(engine, min_confidence=70.0, max_retries=2)
    print("   ✓ ConfidenceRetryEngine created")
    print(f"   Min confidence: {retry_engine.min_confidence}")
    print(f"   Max retries: {retry_engine.max_retries}")
    print(f"   Retry strategies: {len(retry_engine.retry_strategies)}")
except Exception as e:
    print(f"   ✗ Creation failed: {e}")
    sys.exit(1)

# Test 5: Create SmartRetryOrchestrator
print("\n✅ Test 5: Creating SmartRetryOrchestrator...")
try:
    smart_ocr = SmartRetryOrchestrator(
        engine,
        min_confidence=70.0,
        enable_multiscale=True,
        enable_retry=True
    )
    print("   ✓ SmartRetryOrchestrator created")
    print(f"   Multi-scale enabled: {smart_ocr.enable_multiscale}")
    print(f"   Retry enabled: {smart_ocr.enable_retry}")
except Exception as e:
    print(f"   ✗ Creation failed: {e}")
    sys.exit(1)

# Test 6: Check orchestrator integration
print("\n✅ Test 6: Verifying orchestrator integration...")
try:
    from src.orchestrator import run_pipeline
    import inspect
    sig = inspect.signature(run_pipeline)
    params = list(sig.parameters.keys())
    
    if 'use_phase2' in params:
        print("   ✓ Phase 2 parameter integrated into run_pipeline")
    else:
        print("   ✗ Phase 2 parameter NOT found in run_pipeline")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Integration check failed: {e}")
    sys.exit(1)

# Test 7: Check CLI argument
print("\n✅ Test 7: Verifying CLI argument...")
try:
    from src.orchestrator import parse_args
    args = parse_args(['--input', 'test.pdf', '--output', 'test.json', '--phase2'])
    
    if hasattr(args, 'phase2') and args.phase2:
        print("   ✓ --phase2 CLI argument works")
    else:
        print("   ✗ --phase2 CLI argument not found")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ CLI check failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("✅ ALL PHASE 2 VALIDATION TESTS PASSED")
print("="*60)
print("\nPhase 2 is ready to use!")
print("\nTo enable Phase 2 optimizations:")
print("  python -m src.orchestrator --input doc.pdf --output result.json --phase2")
print("\nOr use the quick-start script:")
print("  phase2_quickstart.bat \"path/to/document.pdf\"")
print()

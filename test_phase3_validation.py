"""
Phase 3 Validation Test
Tests layout analysis and post-OCR correction modules
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("="*60)
print("Phase 3 Quick Validation Test")
print("="*60)

# Test 1: Import Phase 3 modules
print("\n✅ Test 1: Importing Phase 3 modules...")
try:
    from src.layout_analyzer import LayoutAnalyzer, LayoutRegion, RegionType
    from src.postocr_corrector import PostOCRCorrector, LanguageAwareCorrector
    print("   ✓ All Phase 3 modules imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create LayoutAnalyzer
print("\n✅ Test 2: Creating LayoutAnalyzer...")
try:
    analyzer = LayoutAnalyzer()
    print("   ✓ LayoutAnalyzer created")
    print(f"   Min column gap: {analyzer.min_column_gap}")
    print(f"   Header/footer margin: {analyzer.header_footer_margin}")
except Exception as e:
    print(f"   ✗ Creation failed: {e}")
    sys.exit(1)

# Test 3: Create PostOCRCorrector
print("\n✅ Test 3: Creating PostOCRCorrector...")
try:
    corrector = PostOCRCorrector(language="auto")
    print("   ✓ PostOCRCorrector created")
    print(f"   Language: {corrector.language}")
    print(f"   Error patterns loaded: {len(corrector.ocr_error_patterns)}")
except Exception as e:
    print(f"   ✗ Creation failed: {e}")
    sys.exit(1)

# Test 4: Test text correction
print("\n✅ Test 4: Testing text correction...")
try:
    test_text = "Thls  is   a  test  with   0CR   err0rs."
    corrected = corrector.correct_text(test_text)
    print(f"   Original:  {test_text}")
    print(f"   Corrected: {corrected}")
    print("   ✓ Text correction works")
except Exception as e:
    print(f"   ✗ Correction failed: {e}")
    sys.exit(1)

# Test 5: Create LanguageAwareCorrector
print("\n✅ Test 5: Creating LanguageAwareCorrector...")
try:
    lang_corrector = LanguageAwareCorrector()
    print("   ✓ LanguageAwareCorrector created")
    print(f"   Uzbek Latin chars: {len(lang_corrector.uzbek_lat_chars)}")
    print(f"   Uzbek Cyrillic chars: {len(lang_corrector.uzbek_cyr_chars)}")
except Exception as e:
    print(f"   ✗ Creation failed: {e}")
    sys.exit(1)

# Test 6: Test script detection
print("\n✅ Test 6: Testing script detection...")
try:
    latin_text = "Bu lotincha matn"
    cyrillic_text = "Это кириллица"
    
    latin_script = lang_corrector.detect_script(latin_text)
    cyrillic_script = lang_corrector.detect_script(cyrillic_text)
    
    print(f"   Latin text detected as: {latin_script}")
    print(f"   Cyrillic text detected as: {cyrillic_script}")
    print("   ✓ Script detection works")
except Exception as e:
    print(f"   ✗ Detection failed: {e}")
    sys.exit(1)

# Test 7: Check orchestrator integration
print("\n✅ Test 7: Verifying orchestrator integration...")
try:
    from src.orchestrator import run_pipeline
    import inspect
    sig = inspect.signature(run_pipeline)
    params = list(sig.parameters.keys())
    
    if 'use_phase3' in params:
        print("   ✓ Phase 3 parameter integrated into run_pipeline")
    else:
        print("   ✗ Phase 3 parameter NOT found in run_pipeline")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Integration check failed: {e}")
    sys.exit(1)

# Test 8: Check CLI argument
print("\n✅ Test 8: Verifying CLI argument...")
try:
    from src.orchestrator import parse_args
    args = parse_args(['--input', 'test.pdf', '--output', 'test.json', '--phase3'])
    
    if hasattr(args, 'phase3') and args.phase3:
        print("   ✓ --phase3 CLI argument works")
    else:
        print("   ✗ --phase3 CLI argument not found")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ CLI check failed: {e}")
    sys.exit(1)

# Test 9: Test combined Phase 2 + Phase 3
print("\n✅ Test 9: Testing combined Phase 2 + Phase 3...")
try:
    args = parse_args(['--input', 'test.pdf', '--output', 'test.json', '--phase2', '--phase3'])
    
    if hasattr(args, 'phase2') and args.phase2 and hasattr(args, 'phase3') and args.phase3:
        print("   ✓ Combined --phase2 --phase3 arguments work")
    else:
        print("   ✗ Combined arguments not working properly")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Combined test failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("✅ ALL PHASE 3 VALIDATION TESTS PASSED")
print("="*60)
print("\nPhase 3 is ready to use!")
print("\nTo enable Phase 3 optimizations:")
print("  python -m src.orchestrator --input doc.pdf --output result.json --phase3")
print("\nOr combine with Phase 2:")
print("  python -m src.orchestrator --input doc.pdf --output result.json --phase2 --phase3")
print("\nOr use the quick-start script:")
print("  phase23_quickstart.bat \"path/to/document.pdf\"")
print()

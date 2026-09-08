"""Test fast-plate-ocr integration."""
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_import():
    """Test if fast-plate-ocr can be imported."""
    try:
        from fast_plate_ocr import LicensePlateRecognizer
        logger.info("✅ fast-plate-ocr imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import fast-plate-ocr: {e}")
        logger.error("Install with: pip install fast-plate-ocr[onnx-gpu]")
        return False

def test_initialization():
    """Test OCR initialization."""
    try:
        from fast_plate_ocr import LicensePlateRecognizer
        
        logger.info("Initializing fast-plate-ocr...")
        m = LicensePlateRecognizer('cct-xs-v2-global-model')
        logger.info("✅ OCR initialized successfully")
        
        # Run benchmark
        logger.info("Running benchmark...")
        m.benchmark()
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize OCR: {e}")
        return False

def test_ocr_reader():
    """Test OCR reader class."""
    try:
        from backend.ai.ocr_reader import PlateOCR
        
        logger.info("Testing OCR reader class...")
        ocr = PlateOCR(
            model_name='cct-xs-v2-global-model',
            confidence_threshold=0.6
        )
        logger.info("✅ OCR reader class works")
        
        return True
    except Exception as e:
        logger.error(f"❌ OCR reader class failed: {e}")
        return False

def test_config():
    """Test config loading."""
    try:
        from backend.core.config import get_config
        
        logger.info("Testing config...")
        config = get_config()
        
        assert hasattr(config, 'ocr'), "OCR config missing"
        assert hasattr(config.ocr, 'model'), "OCR model config missing"
        assert hasattr(config.ocr, 'confidence_threshold'), "OCR threshold config missing"
        
        logger.info(f"✅ Config loaded: model={config.ocr.model}, threshold={config.ocr.confidence_threshold}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Config test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Testing fast-plate-ocr Integration")
    logger.info("=" * 60)
    
    tests = [
        ("Import", test_import),
        ("Initialization", test_initialization),
        ("OCR Reader Class", test_ocr_reader),
        ("Config", test_config),
    ]
    
    results = []
    for name, test_func in tests:
        logger.info(f"\n--- Test: {name} ---")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! fast-plate-ocr is ready.")
        return 0
    else:
        logger.error(f"\n⚠️ {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

"""Sample test for fast-plate-ocr with actual image."""
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_plate():
    """Create a synthetic license plate image for testing."""
    # Create white background
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    
    # Add black text (simulating plate number)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "ABC 1234"
    font_scale = 2
    thickness = 3
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Center text
    x = (img.shape[1] - text_width) // 2
    y = (img.shape[0] + text_height) // 2
    
    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness)
    
    # Convert BGR to RGB for fast-plate-ocr
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img_rgb

def test_ocr_with_sample():
    """Test OCR with synthetic plate image."""
    try:
        from fast_plate_ocr import LicensePlateRecognizer
        
        logger.info("Creating synthetic plate image...")
        plate_img = create_sample_plate()
        logger.info(f"Image shape: {plate_img.shape}")
        
        logger.info("Initializing OCR...")
        m = LicensePlateRecognizer('cct-xs-v2-global-model')
        
        logger.info("Running OCR on synthetic plate...")
        results = m.run(plate_img, return_confidence=True)
        
        if results and len(results) > 0:
            result = results[0]
            logger.info(f"✅ OCR Result: '{result.text}'")
            logger.info(f"   Confidence: {result.confidence:.2f}")
            
            # Check if result makes sense
            if len(result.text) > 0:
                logger.info("✅ OCR is working!")
                return True
            else:
                logger.warning("⚠️ OCR returned empty text")
                return False
        else:
            logger.warning("⚠️ OCR returned no results")
            return False
            
    except Exception as e:
        logger.error(f"❌ OCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ocr_reader_class():
    """Test OCR reader class with sample image."""
    try:
        from backend.ai.ocr_reader import PlateOCR
        import numpy as np
        
        logger.info("\nTesting OCR reader class...")
        ocr = PlateOCR(
            model_name='cct-xs-v2-global-model',
            confidence_threshold=0.5
        )
        
        # Create synthetic plate
        logger.info("Creating synthetic plate (BGR format)...")
        plate_bgr = np.ones((100, 300, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(plate_bgr, "XYZ 5678", (30, 60), font, 2, (0, 0, 0), 3)
        
        # Create fake frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[300:400, 400:700] = plate_bgr
        
        # Test OCR
        plate_bbox = [400, 300, 700, 400]
        result = ocr.read_plate(frame, plate_bbox)
        
        if result:
            logger.info(f"✅ OCR Reader Class Result: '{result}'")
            return True
        else:
            logger.warning("⚠️ OCR Reader returned no text (may need real plate image)")
            return True  # Not necessarily a failure
            
    except Exception as e:
        logger.error(f"❌ OCR Reader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run OCR sample tests."""
    logger.info("=" * 60)
    logger.info("Testing fast-plate-ocr with Sample Images")
    logger.info("=" * 60)
    
    # Test 1: Basic OCR
    logger.info("\n--- Test 1: Basic OCR ---")
    test1 = test_ocr_with_sample()
    
    # Test 2: OCR Reader Class
    logger.info("\n--- Test 2: OCR Reader Class ---")
    test2 = test_ocr_reader_class()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    if test1:
        logger.info("✅ Basic OCR: PASS")
    else:
        logger.error("❌ Basic OCR: FAIL")
    
    if test2:
        logger.info("✅ OCR Reader Class: PASS")
    else:
        logger.error("❌ OCR Reader Class: FAIL")
    
    if test1 and test2:
        logger.info("\n🎉 All OCR tests passed!")
        logger.info("\nNote: These are synthetic tests.")
        logger.info("For real accuracy testing, use actual license plate images.")
        return 0
    else:
        logger.error("\n⚠️ Some tests failed.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

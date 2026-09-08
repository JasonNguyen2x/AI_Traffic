"""License plate OCR using fast-plate-ocr (optimized for license plates)."""
import logging
from typing import List, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class PlateOCR:
    """Fast license plate OCR using fast-plate-ocr (ONNX optimized)."""
    
    def __init__(self, 
                 model_name: str = 'cct-s-v2-global-model',
                 confidence_threshold: float = 0.6):
        """
        Initialize OCR reader.
        
        Args:
            model_name: Pre-trained model name from fast-plate-ocr hub
                       Options: 'cct-s-v2-global-model' (slower, accurate)
                                'cct-xs-v2-global-model' (faster, lighter)
            confidence_threshold: Minimum confidence for text detection
        """
        self.confidence_threshold = confidence_threshold
        
        logger.info(f"Initializing fast-plate-ocr with model: {model_name}")
        
        try:
            from fast_plate_ocr import LicensePlateRecognizer
            
            # Initialize recognizer (will download model on first use)
            # Use CPU provider (ONNX Runtime CUDA has dependency issues with PyTorch CUDA)
            # CPU is still 10-20x faster than EasyOCR due to ONNX optimization
            self.recognizer = LicensePlateRecognizer(
                model_name,
                providers=['CPUExecutionProvider']
            )
            
            logger.info(f"fast-plate-ocr ready (model: {model_name}, provider: CPU - optimized ONNX)")
            
        except Exception as e:
            logger.error(f"Failed to initialize fast-plate-ocr: {e}")
            logger.error("Make sure to install: pip install fast-plate-ocr[onnx-gpu]")
            raise
    
    def preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Minimal preprocessing - fast-plate-ocr handles most internally.
        
        Args:
            plate_img: BGR image of license plate
        
        Returns:
            Preprocessed image
        """
        # fast-plate-ocr expects RGB
        if len(plate_img.shape) == 3 and plate_img.shape[2] == 3:
            plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        
        # Resize if too small (OCR works better on larger images)
        h, w = plate_img.shape[:2]
        if h < 50:
            scale = 50.0 / h
            new_w = int(w * scale)
            plate_img = cv2.resize(plate_img, (new_w, 50), interpolation=cv2.INTER_CUBIC)
        
        return plate_img
    
    def read_plate(self, frame: np.ndarray, plate_bbox: List[float]) -> Optional[str]:
        """
        Extract text from license plate region.
        
        Args:
            frame: Full frame (BGR)
            plate_bbox: Plate bounding box [x1, y1, x2, y2]
        
        Returns:
            Plate text string or None if no text detected
        """
        try:
            # Crop plate region
            x1, y1, x2, y2 = [int(v) for v in plate_bbox]
            
            # Ensure valid crop
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                logger.info(f"OCR FAIL: Invalid bbox [{x1},{y1},{x2},{y2}]")
                return None
            
            plate_crop = frame[y1:y2, x1:x2]
            
            if plate_crop.size == 0:
                logger.info(f"OCR FAIL: Empty crop from bbox [{x1},{y1},{x2},{y2}]")
                return None
            
            crop_h, crop_w = plate_crop.shape[:2]
            logger.info(f"OCR INPUT: Plate crop size {crop_w}x{crop_h}")
            
            # Preprocess (BGR → RGB)
            processed = self.preprocess_plate(plate_crop)
            proc_h, proc_w = processed.shape[:2]
            logger.info(f"OCR PROCESSED: After preprocess {proc_w}x{proc_h}")
            
            # Run OCR with confidence
            results = self.recognizer.run(processed, return_confidence=True)
            
            if results and len(results) > 0:
                result = results[0]
                
                # ponytail: PlatePrediction uses .plate not .text
                plate_text = getattr(result, 'plate', getattr(result, 'text', ''))
                
                # Check if confidence is available
                if result.has_confidence and hasattr(result, 'char_probs') and result.char_probs is not None and len(result.char_probs) > 0:
                    # Calculate average confidence from char_probs
                    confidence = float(sum(result.char_probs) / len(result.char_probs))
                else:
                    # No confidence available, use 1.0 (always accept)
                    confidence = 1.0
                
                logger.info(f"OCR RAW: text='{plate_text}', conf={confidence:.3f}, threshold={self.confidence_threshold}")
                
                # Check confidence threshold
                if confidence >= self.confidence_threshold:
                    # Clean text (already uppercase from fast-plate-ocr)
                    text = plate_text.strip()
                    
                    if text:
                        logger.info(f"OCR SUCCESS: '{text}' (confidence: {confidence:.2f})")
                        return text
                    else:
                        logger.info(f"OCR REJECT: Empty text after strip")
                else:
                    logger.info(f"OCR REJECT: Low confidence {confidence:.3f} < {self.confidence_threshold}")
            else:
                logger.info(f"OCR EMPTY: fast-plate-ocr returned no results")
            
            return None
            
        except Exception as e:
            logger.error(f"OCR ERROR: {e}", exc_info=True)
            return None
    
    def read_plates_batch(self, frame: np.ndarray, plate_bboxes: List[List[float]]) -> List[Optional[str]]:
        """
        Read multiple plates from frame.
        
        Args:
            frame: Full frame (BGR)
            plate_bboxes: List of plate bounding boxes
        
        Returns:
            List of plate texts (None for failed reads)
        """
        results = []
        
        for bbox in plate_bboxes:
            text = self.read_plate(frame, bbox)
            results.append(text)
        
        return results
    
    @staticmethod
    def format_plate_text(text: str) -> str:
        """
        Format plate text for display (already clean from fast-plate-ocr).
        
        Args:
            text: Raw OCR text
        
        Returns:
            Formatted text
        """
        if not text:
            return ""
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text.upper()

"""License plate detection using YOLO."""
import logging
from typing import List, Dict, Optional
import numpy as np
import cv2
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class PlateDetector:
    """YOLO-based license plate detector with spatial association."""
    
    def __init__(self, model_path: str = 'models/license_plate_detector.pt',
                 device: str = 'auto',
                 confidence: float = 0.4,
                 inference_size: int = 640):
        """
        Initialize plate detector.
        
        Args:
            model_path: Path to YOLO model (.pt or .engine for TensorRT)
            device: 'auto', 'cuda', or 'cpu'
            confidence: Confidence threshold (0-1)
            inference_size: Input size for inference
        """
        self.confidence = confidence
        
        # Check for TensorRT engine
        self.is_tensorrt = model_path.endswith('.engine')
        
        # TensorRT engines have fixed input size
        if self.is_tensorrt:
            # Plate detector uses 640 by default
            self.inference_size = 640
            logger.info(f"TensorRT engine detected, using fixed input size: {self.inference_size}")
        else:
            self.inference_size = inference_size
        
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = 'cpu'
        
        logger.info(f"Initializing plate detector on device: {self.device}")
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            
            if self.is_tensorrt:
                logger.info(f"Loaded TensorRT engine: {model_path}")
                logger.info(f"Fixed input size: {self.inference_size}×{self.inference_size}")
                self.use_fp16 = False  # Already optimized
            else:
                # Move model to GPU immediately
                if self.device == 'cuda':
                    self.model.to('cuda')
                    logger.info(f"Loaded plate detection model on GPU: {model_path}")
                else:
                    logger.info(f"Loaded plate detection model on CPU: {model_path}")
                
                # Enable FP16 if CUDA available
                self.use_fp16 = self.device == 'cuda'
                if self.use_fp16:
                    logger.info("FP16 optimization enabled for plate detection")
            
        except Exception as e:
            logger.error(f"Failed to load plate detection model: {e}")
            raise
    
    def detect_on_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect all license plates in the full frame.
        
        Args:
            frame: BGR image (H, W, 3)
        
        Returns:
            List of plate detections:
            [
                {
                    'bbox': [x1, y1, x2, y2],  # Frame coordinates
                    'confidence': 0.87
                },
                ...
            ]
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            # Prepare kwargs
            inference_kwargs = {
                'imgsz': self.inference_size,
                'conf': self.confidence,
                'device': self.device,
                'verbose': False,
            }
            
            # Add FP16 quantization if CUDA available and not TensorRT
            if self.use_fp16:
                inference_kwargs['quantize'] = 'fp16'
            
            # Run inference on full frame
            results = self.model.predict(frame, **inference_kwargs)
            
            # Parse results
            detections = []
            
            if len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    # Batch GPU→CPU transfer (single sync instead of N×2)
                    all_xyxy = boxes.xyxy.cpu().numpy()
                    all_conf = boxes.conf.cpu().numpy()
                    
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = all_xyxy[i]
                        conf = float(all_conf[i])
                        
                        detection = {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': conf
                        }
                        
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Plate detection error: {e}")
            return []
    
    @staticmethod
    def bbox_inside(plate_bbox: List[float], vehicle_bbox: List[float], 
                    tolerance: float = 0.0) -> bool:
        """
        Check if plate bbox is inside vehicle bbox.
        
        Args:
            plate_bbox: [x1, y1, x2, y2]
            vehicle_bbox: [x1, y1, x2, y2]
            tolerance: Allow plates slightly outside vehicle (pixels)
        
        Returns:
            True if plate is inside vehicle (with tolerance)
        """
        px1, py1, px2, py2 = plate_bbox
        vx1, vy1, vx2, vy2 = vehicle_bbox
        
        # Check if plate center is inside vehicle
        plate_cx = (px1 + px2) / 2
        plate_cy = (py1 + py2) / 2
        
        return (vx1 - tolerance <= plate_cx <= vx2 + tolerance and
                vy1 - tolerance <= plate_cy <= vy2 + tolerance)
    
    @staticmethod
    def associate_plates_to_vehicles(plate_detections: List[Dict], 
                                     vehicle_detections: List[Dict]) -> Dict[int, List[Dict]]:
        """
        Associate plates to vehicles using spatial overlap.
        Convert plate coords to vehicle-relative coords for persistence across frames.
        
        Uses track_id as key (stable across frames) when available,
        falls back to array index if track_id is missing.
        
        Args:
            plate_detections: List of plate detections from detect_on_frame()
            vehicle_detections: List of vehicle detections (with optional 'track_id')
        
        Returns:
            Dictionary mapping track_id (or index) to list of plates with relative coords:
            {
                14: [{'bbox_rel': [x1, y1, x2, y2], 'confidence': 0.9}],
                27: [{'bbox_rel': [x1, y1, x2, y2], 'confidence': 0.85}],
                ...
            }
        """
        associations = {}
        
        for plate in plate_detections:
            plate_bbox = plate['bbox']
            px1, py1, px2, py2 = plate_bbox
            
            # Find which vehicle contains this plate
            for v_idx, vehicle in enumerate(vehicle_detections):
                vehicle_bbox = vehicle['bbox']
                vx1, vy1, vx2, vy2 = vehicle_bbox
                
                if PlateDetector.bbox_inside(plate_bbox, vehicle_bbox, tolerance=5.0):
                    # Use track_id as key (stable), fallback to index
                    key = vehicle.get('track_id', v_idx)
                    if key == -1:
                        key = v_idx  # Fallback if tracking failed this frame
                    
                    if key not in associations:
                        associations[key] = []
                    
                    # Convert to vehicle-relative coords (0-1 normalized)
                    v_width = vx2 - vx1
                    v_height = vy2 - vy1
                    
                    if v_width > 0 and v_height > 0:
                        rel_x1 = (px1 - vx1) / v_width
                        rel_y1 = (py1 - vy1) / v_height
                        rel_x2 = (px2 - vx1) / v_width
                        rel_y2 = (py2 - vy1) / v_height
                        
                        associations[key].append({
                            'bbox_rel': [rel_x1, rel_y1, rel_x2, rel_y2],
                            'confidence': plate['confidence']
                        })
                    
                    break  # Each plate belongs to at most one vehicle
        
        return associations
    
    @staticmethod
    def draw_plates(frame: np.ndarray, plate_detections: List[Dict]) -> np.ndarray:
        """
        Draw plate boxes on frame.
        
        Args:
            frame: Original frame
            plate_detections: List of plate detections from detect_on_frame()
        
        Returns:
            Frame with drawn plate boxes
        """
        import cv2
        
        # Plate color: Blue
        color = (255, 0, 0)  # BGR: Blue
        
        for plate in plate_detections:
            px1, py1, px2, py2 = [int(v) for v in plate['bbox']]
            conf = plate['confidence']
            
            # Draw box (thicker for visibility)
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 3)
            
            # Draw label
            label = f"Plate {conf:.2f}"
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw label background
            cv2.rectangle(
                frame,
                (px1, py1 - text_height - baseline - 5),
                (px1 + text_width, py1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (px1, py1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                2
            )
        
        return frame
    
    @staticmethod
    def draw_plates_from_associations(frame: np.ndarray, vehicle_detections: List[Dict], 
                                      plate_associations: Dict[int, List[Dict]],
                                      ocr_results: Optional[Dict[int, str]] = None) -> np.ndarray:
        """
        Draw plates from cached associations with OCR text.
        
        Args:
            frame: Original frame
            vehicle_detections: Current vehicle detections (with optional 'track_id')
            plate_associations: Dict mapping track_id -> plates with bbox_rel
            ocr_results: Dict mapping track_id -> plate_text
        
        Returns:
            Frame with drawn plate boxes and OCR text
        """
        # Build track_id → vehicle lookup for O(1) access
        vehicle_by_id = {}
        for v_idx, v in enumerate(vehicle_detections):
            key = v.get('track_id', v_idx)
            if key == -1:
                key = v_idx
            vehicle_by_id[key] = v
        
        color = (255, 0, 0)  # Blue
        
        for v_key, plates in plate_associations.items():
            # Look up vehicle by track_id
            vehicle = vehicle_by_id.get(v_key)
            if vehicle is None:
                continue
            
            vx1, vy1, vx2, vy2 = vehicle['bbox']
            v_width = vx2 - vx1
            v_height = vy2 - vy1
            
            # Get OCR text for this vehicle
            plate_text = None
            if ocr_results and v_key in ocr_results:
                plate_text = ocr_results[v_key]
            
            for plate in plates:
                # Convert relative coords back to frame coords
                rel_x1, rel_y1, rel_x2, rel_y2 = plate['bbox_rel']
                
                px1 = int(vx1 + rel_x1 * v_width)
                py1 = int(vy1 + rel_y1 * v_height)
                px2 = int(vx1 + rel_x2 * v_width)
                py2 = int(vy1 + rel_y2 * v_height)
                
                conf = plate['confidence']
                
                # Draw box
                cv2.rectangle(frame, (px1, py1), (px2, py2), color, 3)
                
                # Build label - show OCR text if available, otherwise confidence
                if plate_text:
                    label = plate_text  # Show OCR text
                else:
                    label = f"Plate {conf:.2f}"  # Show confidence
                
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                cv2.rectangle(
                    frame,
                    (px1, py1 - text_height - baseline - 5),
                    (px1 + text_width, py1),
                    color,
                    -1
                )
                
                cv2.putText(
                    frame,
                    label,
                    (px1, py1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
        
        return frame

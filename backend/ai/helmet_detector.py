"""Helmet detection for motorcycles using YOLO."""
import logging
from typing import List, Dict, Optional
import numpy as np
import cv2
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class HelmetDetector:
    """YOLO-based helmet detector with spatial association to motorcycles."""
    
    def __init__(self, model_path: str = 'models/helmet_detector.pt',
                 device: str = 'auto',
                 confidence: float = 0.5,
                 inference_size: int = 640):
        """
        Initialize helmet detector.
        
        Args:
            model_path: Path to YOLO model (.pt or .engine for TensorRT)
            device: 'auto', 'cuda', or 'cpu'
            confidence: Confidence threshold (0-1)
            inference_size: Input size for inference
        """
        self.confidence = confidence
        
        # Check for TensorRT engine
        self.is_tensorrt = model_path.endswith('.engine')
        
        if self.is_tensorrt:
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
        
        logger.info(f"Initializing helmet detector on device: {self.device}")
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            
            if self.is_tensorrt:
                logger.info(f"Loaded TensorRT engine: {model_path}")
                self.use_fp16 = False
            else:
                if self.device == 'cuda':
                    self.model.to('cuda')
                    logger.info(f"Loaded helmet detection model on GPU: {model_path}")
                else:
                    logger.info(f"Loaded helmet detection model on CPU: {model_path}")
                
                self.use_fp16 = self.device == 'cuda'
                if self.use_fp16:
                    logger.info("FP16 optimization enabled for helmet detection")
            
        except Exception as e:
            logger.error(f"Failed to load helmet detection model: {e}")
            raise
    
    def detect_on_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect all helmets in the full frame.
        
        Single inference on full frame - O(1) instead of O(N) motorcycles.
        
        Args:
            frame: BGR image
            
        Returns:
            List of helmet detections: [{'bbox': [x1, y1, x2, y2], 'confidence': 0.87, 'class': 'helmet'}, ...]
        """
        results = self.model.predict(
            frame,
            imgsz=self.inference_size,
            conf=self.confidence,
            device=self.device,
            half=self.use_fp16,
            verbose=False
        )
        
        helmets = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.cpu().numpy()
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    # Get class name (usually 'helmet' or 'no-helmet')
                    cls_name = result.names[cls_id] if hasattr(result, 'names') else str(cls_id)
                    
                    helmets.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': conf,
                        'class': cls_name
                    })
        
        return helmets
    
    @staticmethod
    def bbox_inside(helmet_bbox: List[float], motorcycle_bbox: List[float],
                    tolerance: float = 5.0) -> bool:
        """
        Check if helmet bbox is inside motorcycle bbox (with tolerance).
        
        Uses center point of helmet for robust matching.
        
        Args:
            helmet_bbox: [x1, y1, x2, y2]
            motorcycle_bbox: [x1, y1, x2, y2]
            tolerance: Pixels to expand motorcycle bbox
            
        Returns:
            True if helmet center is inside motorcycle bbox
        """
        hx1, hy1, hx2, hy2 = helmet_bbox
        mx1, my1, mx2, my2 = motorcycle_bbox
        
        # Calculate helmet center
        helmet_cx = (hx1 + hx2) / 2
        helmet_cy = (hy1 + hy2) / 2
        
        # Expand motorcycle bbox with tolerance
        mx1 -= tolerance
        my1 -= tolerance
        mx2 += tolerance
        my2 += tolerance
        
        # Check if center is inside
        return mx1 <= helmet_cx <= mx2 and my1 <= helmet_cy <= my2
    
    @staticmethod
    def associate_helmets_to_motorcycles(helmet_detections: List[Dict],
                                         vehicle_detections: List[Dict],
                                         tolerance: float = 10.0) -> Dict[int, Dict]:
        """
        Associate helmet status to motorcycles.
        
        Model classes:
          - BikeWithRider (0): Motorcycle with rider
          - NoHelmet (1): Rider without helmet
          - Helmet (2): Rider with helmet
        
        Logic:
          1. Match vehicle detector's motorcycle with BikeWithRider bbox
          2. Check if Helmet or NoHelmet exists inside BikeWithRider
          3. Return helmet status per motorcycle
        
        Args:
            helmet_detections: List of detections from helmet model (BikeWithRider, Helmet, NoHelmet)
            vehicle_detections: List of vehicle detections from vehicle detector
            tolerance: IoU tolerance for matching
            
        Returns:
            Dict keyed by vehicle index: {
                0: {'has_helmet': True, 'confidence': 0.92, 'helmet_class': 'Helmet'},
                3: {'has_helmet': False, 'confidence': 0.85, 'helmet_class': 'NoHelmet'},
            }
        """
        associations = {}
        
        # Separate detections by class
        bikes_with_rider = [d for d in helmet_detections if d['class'] == 'BikeWithRider']
        helmets = [d for d in helmet_detections if d['class'] == 'Helmet']
        no_helmets = [d for d in helmet_detections if d['class'] == 'NoHelmet']
        
        for v_idx, vehicle in enumerate(vehicle_detections):
            # Only process motorcycles
            if vehicle['class'] != 'motorcycle':
                continue
            
            v_bbox = vehicle['bbox']
            vx1, vy1, vx2, vy2 = v_bbox
            
            # Find matching BikeWithRider (IoU overlap)
            best_bike_match = None
            best_iou = 0.3  # Minimum IoU threshold
            
            for bike in bikes_with_rider:
                iou = HelmetDetector.compute_iou(v_bbox, bike['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_bike_match = bike
            
            if not best_bike_match:
                # No BikeWithRider detected for this motorcycle, skip
                continue
            
            bike_bbox = best_bike_match['bbox']
            
            # Check for Helmet inside BikeWithRider bbox
            has_helmet = False
            helmet_conf = 0.0
            helmet_class = 'Unknown'
            helmet_bbox = None  # Store actual helmet bbox for drawing
            
            for helmet in helmets:
                if HelmetDetector.bbox_inside(helmet['bbox'], bike_bbox, tolerance):
                    has_helmet = True
                    helmet_conf = helmet['confidence']
                    helmet_class = 'Helmet'
                    helmet_bbox = helmet['bbox']  # Save bbox
                    logger.debug(f"Helmet found: bbox={helmet_bbox}, conf={helmet_conf:.2f}")
                    break
            
            # If no helmet found, check for NoHelmet
            if not has_helmet:
                for no_helmet in no_helmets:
                    if HelmetDetector.bbox_inside(no_helmet['bbox'], bike_bbox, tolerance):
                        has_helmet = False
                        helmet_conf = no_helmet['confidence']
                        helmet_class = 'NoHelmet'
                        helmet_bbox = no_helmet['bbox']  # Save bbox
                        logger.debug(f"NoHelmet found: bbox={helmet_bbox}, conf={helmet_conf:.2f}")
                        break
            
            # Store association
            associations[v_idx] = {
                'has_helmet': has_helmet,
                'confidence': helmet_conf,
                'helmet_class': helmet_class,
                'helmet_bbox': helmet_bbox  # Actual helmet/nohelmet bbox for drawing
            }
        
        return associations
    
    @staticmethod
    def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """
        Compute IoU (Intersection over Union) between two bboxes.
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            
        Returns:
            IoU score (0-1), or 0.0 if inputs are invalid
        """
        # Guard against None or invalid inputs
        if bbox1 is None or bbox2 is None:
            return 0.0
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        inter_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
        
    
    @staticmethod
    def draw_helmets_from_associations(frame: np.ndarray,
                                       vehicle_detections: List[Dict],
                                       helmet_associations: Dict[int, Dict]) -> np.ndarray:
        """
        Draw helmet status on motorcycles from cached associations.
        
        Args:
            frame: BGR image to draw on
            vehicle_detections: List of vehicle detections
            helmet_associations: Dict from associate_helmets_to_motorcycles()
            
        Returns:
            Frame with helmet status drawn
        """
        for v_idx, vehicle in enumerate(vehicle_detections):
            if vehicle['class'] != 'motorcycle':
                continue
            
            if v_idx not in helmet_associations:
                continue
            
            assoc = helmet_associations[v_idx]
            v_bbox = vehicle['bbox']
            vx1, vy1, vx2, vy2 = [int(c) for c in v_bbox]
            
            # Draw actual Helmet/NoHelmet bbox if available
            if assoc['helmet_bbox'] is not None:
                hx1, hy1, hx2, hy2 = [int(c) for c in assoc['helmet_bbox']]
                
                # Debug: Log bbox coordinates to verify they're different from motorcycle
                logger.debug(f"Drawing {assoc['helmet_class']}: helmet_bbox=[{hx1},{hy1},{hx2},{hy2}], motorcycle_bbox=[{vx1},{vy1},{vx2},{vy2}]")
                
                # Color: Yellow for Helmet, Red for NoHelmet
                color = (0, 255, 255) if assoc['has_helmet'] else (0, 0, 255)  # BGR: Yellow = (0,255,255)
                
                # Draw helmet bbox (actual helmet/nohelmet detection, NOT motorcycle)
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)
                
                # Label with class and confidence
                label = f"{assoc['helmet_class']} {assoc['confidence']:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = max(hy1 - 5, label_size[1])
                
                # Background
                cv2.rectangle(frame,
                            (hx1, label_y - label_size[1] - 4),
                            (hx1 + label_size[0] + 4, label_y + 2),
                            color, -1)
                
                # Text
                cv2.putText(frame, label, (hx1 + 2, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # Black text for visibility
        
        return frame

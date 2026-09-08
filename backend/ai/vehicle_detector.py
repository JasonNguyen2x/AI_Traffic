"""Vehicle detection using YOLO."""
import logging
from typing import List, Dict, Optional
import numpy as np
import cv2
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class VehicleDetector:
    """YOLO-based vehicle detector."""
    
    # COCO class IDs for vehicles
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }
    
    def __init__(self, model_path: str = 'models/yolov11n.pt', 
                 device: str = 'auto',
                 confidence: float = 0.4,
                 inference_size: int = 640):
        """
        Initialize vehicle detector.
        
        Args:
            model_path: Path to YOLO model (.pt or .engine for TensorRT)
            device: 'auto', 'cuda', or 'cpu'
            confidence: Confidence threshold (0-1)
            inference_size: Input size for inference (e.g., 640 or 1280)
        """
        self.confidence = confidence
        
        # Check for TensorRT engine
        self.is_tensorrt = model_path.endswith('.engine')
        
        # TensorRT engines have fixed input size - override if needed
        if self.is_tensorrt:
            # Extract size from filename or use 1280 as default
            if '1280' in model_path or 'yolo11n.engine' in model_path:
                self.inference_size = 1280
                logger.info(f"TensorRT engine detected, using fixed input size: {self.inference_size}")
            else:
                self.inference_size = inference_size
        else:
            self.inference_size = inference_size
        
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Check CUDA availability
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = 'cpu'
        
        logger.info(f"Initializing vehicle detector on device: {self.device}")
        
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
                    logger.info(f"Loaded YOLO model on GPU: {model_path}")
                else:
                    logger.info(f"Loaded YOLO model on CPU: {model_path}")
                
                # Enable FP16 if CUDA available
                self.use_fp16 = self.device == 'cuda'
                if self.use_fp16:
                    logger.info("FP16 optimization enabled")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in a frame.
        
        Args:
            frame: BGR image (H, W, 3)
        
        Returns:
            List of detections:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 0.94,
                    'class': 'car',
                    'class_id': 2
                },
                ...
            ]
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            # Prepare kwargs for inference
            inference_kwargs = {
                'imgsz': self.inference_size,
                'conf': self.confidence,
                'device': self.device,
                'verbose': False,
                'classes': list(self.VEHICLE_CLASSES.keys()),  # Only detect vehicles
            }
            
            # Add FP16 quantization if CUDA available and not TensorRT
            if self.use_fp16:
                inference_kwargs['quantize'] = 'fp16'
            
            # Run inference
            results = self.model.predict(frame, **inference_kwargs)
            
            # Parse results
            detections = []
            
            if len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    # Batch GPU→CPU transfer (single sync instead of N×3)
                    all_xyxy = boxes.xyxy.cpu().numpy()
                    all_conf = boxes.conf.cpu().numpy()
                    all_cls = boxes.cls.cpu().numpy().astype(int)
                    
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = all_xyxy[i]
                        conf = float(all_conf[i])
                        cls_id = int(all_cls[i])
                        cls_name = self.VEHICLE_CLASSES.get(cls_id, 'unknown')
                        
                        detection = {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': conf,
                            'class': cls_name,
                            'class_id': cls_id
                        }
                        
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def track(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect and track vehicles in a frame using ByteTrack.
        
        Returns detections with persistent track_id across frames.
        
        Args:
            frame: BGR image (H, W, 3)
        
        Returns:
            List of detections with tracking:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 0.94,
                    'class': 'car',
                    'class_id': 2,
                    'track_id': 14
                },
                ...
            ]
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            # Prepare kwargs for tracking
            tracking_kwargs = {
                'imgsz': self.inference_size,
                'conf': self.confidence,
                'device': self.device,
                'verbose': False,
                'classes': list(self.VEHICLE_CLASSES.keys()),
                'persist': True,  # Maintain tracking state across frames
                'tracker': 'bytetrack.yaml',  # Use ByteTrack algorithm
            }
            
            # Run tracking (replaces model.predict)
            results = self.model.track(frame, **tracking_kwargs)
            
            # Parse results
            detections = []
            
            if len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    # Batch GPU→CPU transfer
                    all_xyxy = boxes.xyxy.cpu().numpy()
                    all_conf = boxes.conf.cpu().numpy()
                    all_cls = boxes.cls.cpu().numpy().astype(int)
                    
                    # Track IDs (may be None on first frame or if tracking fails)
                    all_ids = None
                    if boxes.id is not None:
                        all_ids = boxes.id.cpu().numpy().astype(int)
                    
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = all_xyxy[i]
                        conf = float(all_conf[i])
                        cls_id = int(all_cls[i])
                        cls_name = self.VEHICLE_CLASSES.get(cls_id, 'unknown')
                        track_id = int(all_ids[i]) if all_ids is not None else -1
                        
                        detection = {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': conf,
                            'class': cls_name,
                            'class_id': cls_id,
                            'track_id': track_id,
                        }
                        
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Tracking error: {e}")
            return []
    
    def reset_tracker(self):
        """Reset ByteTrack state (call when stream disconnects/reconnects)."""
        try:
            if hasattr(self.model, 'predictor') and self.model.predictor is not None:
                # Clear predictor entirely so YOLO reinitializes tracker from scratch.
                # Setting trackers=[] would leave a poisoned state where YOLO
                # thinks the tracker exists but trackers[0] raises IndexError.
                self.model.predictor = None
                logger.info("ByteTrack tracker state reset (predictor cleared)")
            else:
                logger.info("Tracker already clean, no reset needed")
        except Exception as e:
            logger.warning(f"Failed to reset tracker: {e}")
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict], 
                       use_yolo_plot: bool = False, results=None) -> np.ndarray:
        """
        Draw detection boxes on frame.
        
        Args:
            frame: BGR image
            detections: List of detections from detect() or track()
            use_yolo_plot: Use YOLO's built-in plot (faster)
            results: YOLO results object (if use_yolo_plot=True)
        
        Returns:
            Frame with drawn boxes
        """
        # Option 1: Use YOLO's built-in rendering (faster)
        if use_yolo_plot and results is not None:
            try:
                # YOLO has optimized C++ rendering
                return results[0].plot(
                    conf=True,
                    line_width=2,
                    font_size=12,
                    labels=True,
                    boxes=True
                )
            except:
                pass  # Fall back to manual drawing
        
        # Option 2: Manual drawing with OpenCV (more control)
        # Draw directly on frame (no copy) — caller is done reading it
        
        # Colors for different vehicle types
        colors = {
            'car': (0, 255, 0),        # Green
            'motorcycle': (255, 255, 0),  # Cyan
            'bus': (0, 165, 255),      # Orange
            'truck': (0, 0, 255)       # Red
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cls_name = det['class']
            conf = det['confidence']
            
            # Get color
            color = colors.get(cls_name, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label (keep current format without track_id)
            label = f"{cls_name} {conf:.2f}"
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw label background
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )
        
        return frame
    
    def get_device_info(self) -> Dict:
        """Get device information."""
        info = {
            'device': self.device,
            'cuda_available': torch.cuda.is_available(),
            'fp16_enabled': self.use_fp16
        }
        
        if torch.cuda.is_available():
            info['cuda_device'] = torch.cuda.get_device_name(0)
            info['cuda_memory'] = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        
        return info

"""Video file adapter for testing with local files."""
import cv2
import logging
from typing import Optional, Dict
import numpy as np
from pathlib import Path
from backend.stream.base import BaseStreamAdapter

logger = logging.getLogger(__name__)


class VideoAdapter(BaseStreamAdapter):
    """Video file adapter for local file testing."""
    
    def __init__(self, file_path: str):
        """
        Initialize with video file path.
        
        Args:
            file_path: Path to local video file
        """
        super().__init__(url=file_path)
        self.file_path = file_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.loop = True  # Loop video for testing
    
    def connect(self) -> bool:
        """Open video file."""
        if self._connected:
            logger.warning("Already connected to video file")
            return True
        
        # Check if file exists
        if not Path(self.file_path).exists():
            logger.error(f"Video file not found: {self.file_path}")
            return False
        
        logger.info(f"Opening video file: {self.file_path}")
        
        try:
            self.cap = cv2.VideoCapture(self.file_path)
            
            if not self.cap.isOpened():
                logger.error("Failed to open video file")
                return False
            
            # Try to read one frame to verify
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error("Failed to read initial frame from video file")
                self.disconnect()
                return False
            
            # Reset to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            self._connected = True
            info = self.get_stream_info()
            logger.info(f"Video file opened: {info['width']}x{info['height']} @ {info['fps']} FPS")
            logger.info(f"Total frames: {int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
            return True
            
        except Exception as e:
            logger.error(f"Video file open failed: {e}")
            self.disconnect()
            return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read frame from video file."""
        if not self._connected or self.cap is None:
            return None
        
        try:
            ret, frame = self.cap.read()
            
            # If end of video, loop back to start
            if not ret or frame is None:
                if self.loop:
                    logger.info("End of video, looping back to start")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    
                    if not ret or frame is None:
                        logger.warning("Failed to loop video")
                        self._connected = False
                        return None
                else:
                    logger.info("End of video reached")
                    self._connected = False
                    return None
            
            return frame  # BGR format
            
        except Exception as e:
            logger.error(f"Error reading video frame: {e}")
            self._connected = False
            return None
    
    def disconnect(self) -> None:
        """Close video file."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self._connected = False
        logger.info("Video file closed")
    
    def get_stream_info(self) -> Dict:
        """Get video file information."""
        if not self._connected or self.cap is None:
            return {
                "width": 0,
                "height": 0,
                "fps": 0,
                "codec": "unknown",
                "total_frames": 0
            }
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get codec (FourCC)
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        return {
            "width": width,
            "height": height,
            "fps": fps if fps > 0 else 30,
            "codec": codec.strip() if codec.strip() else "unknown",
            "total_frames": total_frames
        }

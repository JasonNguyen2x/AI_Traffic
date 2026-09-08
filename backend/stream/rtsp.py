"""RTSP stream adapter using OpenCV."""
import cv2
import logging
from typing import Optional, Dict
import numpy as np
from urllib.parse import urlparse
from backend.stream.base import BaseStreamAdapter

logger = logging.getLogger(__name__)


class RTSPAdapter(BaseStreamAdapter):
    """RTSP stream adapter."""
    
    def __init__(self, url: str):
        """
        Initialize RTSP adapter with full URL.
        
        Args:
            url: Full RTSP URL (e.g., rtsp://username:password@host:port/path)
        """
        super().__init__(url)
        self.cap: Optional[cv2.VideoCapture] = None
    
    def connect(self) -> bool:
        """Connect to RTSP stream."""
        if self._connected:
            logger.warning("Already connected to RTSP stream")
            return True
        
        # Mask password in logs for security
        log_url = self._mask_password(self.url)
        logger.info(f"Connecting to RTSP stream: {log_url}")
        
        try:
            # Suppress FFmpeg decoder warnings (corrupted frames are auto-skipped)
            import os
            os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'  # AV_LOG_QUIET
            
            # OpenCV VideoCapture with optimized flags
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            
            # Error resilience: continue on corrupted frames
            # ponytail: OpenCV already skips bad frames, these just reduce noise
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Try to enable error concealment (may not work on all builds)
            try:
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            except:
                pass
            
            # Try to read one frame to verify connection
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error("Failed to read initial frame from RTSP stream")
                self.disconnect()
                return False
            
            self._connected = True
            info = self.get_stream_info()
            logger.info(f"RTSP connected: {info['width']}x{info['height']} @ {info['fps']} FPS")
            return True
            
        except Exception as e:
            logger.error(f"RTSP connection failed: {e}")
            self.disconnect()
            return False
    
    @staticmethod
    def _mask_password(url: str) -> str:
        """Mask password in URL for logging."""
        try:
            parsed = urlparse(url)
            if parsed.password:
                masked_netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
                return url.replace(parsed.netloc, masked_netloc)
        except:
            pass
        return url
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read frame from RTSP stream."""
        if not self._connected or self.cap is None:
            return None
        
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame from RTSP stream")
                self._connected = False
                return None
            
            # Validate frame is not corrupted (basic sanity check)
            # ponytail: OpenCV already skips corrupt frames, this catches edge cases
            if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                logger.debug("Received empty/corrupted frame, skipping")
                return None
            
            return frame  # BGR format
            
        except Exception as e:
            logger.error(f"Error reading RTSP frame: {e}")
            self._connected = False
            return None
    
    def disconnect(self) -> None:
        """Disconnect from RTSP stream."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self._connected = False
        logger.info("RTSP stream disconnected")
    
    def get_stream_info(self) -> Dict:
        """Get RTSP stream information."""
        if not self._connected or self.cap is None:
            return {
                "width": 0,
                "height": 0,
                "fps": 0,
                "codec": "unknown"
            }
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Get codec (FourCC)
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        return {
            "width": width,
            "height": height,
            "fps": fps if fps > 0 else 30,  # Default to 30 if unknown
            "codec": codec.strip() if codec.strip() else "H264"
        }

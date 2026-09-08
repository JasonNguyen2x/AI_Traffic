"""HLS stream adapter using OpenCV."""
import cv2
import logging
from typing import Optional, Dict
import numpy as np
from backend.stream.base import BaseStreamAdapter

logger = logging.getLogger(__name__)


class HLSAdapter(BaseStreamAdapter):
    """HLS (HTTP Live Streaming) adapter."""
    
    def __init__(self, url: str):
        """
        Initialize HLS adapter with full URL.
        
        Args:
            url: Full HLS URL (e.g., http://host:port/path/stream.m3u8)
                 If URL doesn't end with .m3u8, common playlist names will be tried.
        """
        super().__init__(url)
        self.cap: Optional[cv2.VideoCapture] = None
        self._normalize_url()
    
    def _normalize_url(self):
        """Normalize HLS URL to ensure it points to .m3u8 playlist."""
        # If URL doesn't end with .m3u8, try common playlist names
        if not self.url.endswith('.m3u8'):
            # Remove trailing slash if present
            base_url = self.url.rstrip('/')
            
            # Common HLS playlist filenames to try
            self.playlist_urls = [
                f"{base_url}/playlist.m3u8",
                f"{base_url}/index.m3u8",
                f"{base_url}/stream.m3u8",
                f"{base_url}/master.m3u8",
                f"{base_url}.m3u8", 
            ]
            
            logger.info(f"HLS URL doesn't end with .m3u8, will try common playlist names")
        else:
            # URL already points to .m3u8 file
            self.playlist_urls = [self.url]
    
    def connect(self) -> bool:
        """Connect to HLS stream. Tries multiple playlist URLs if needed."""
        if self._connected:
            logger.warning("Already connected to HLS stream")
            return True
        
        last_error = None
        
        # Try each possible playlist URL
        for playlist_url in self.playlist_urls:
            logger.info(f"Attempting to connect to HLS: {playlist_url}")
            
            try:
                # Suppress FFmpeg decoder warnings (corrupted frames are auto-skipped)
                import os
                os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'  # AV_LOG_QUIET
                
                # OpenCV can handle HLS via FFmpeg backend
                self.cap = cv2.VideoCapture(playlist_url, cv2.CAP_FFMPEG)
                
                # Set buffer size to 1 (latest frame strategy)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Verify connection by reading one frame
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # Success!
                    self._connected = True
                    self.url = playlist_url  # Update to working URL
                    info = self.get_stream_info()
                    logger.info(f"HLS connected: {playlist_url}")
                    logger.info(f"Stream info: {info['width']}x{info['height']} @ {info['fps']} FPS")
                    return True
                else:
                    # Failed to read frame, try next URL
                    logger.debug(f"Failed to read frame from {playlist_url}")
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                    
            except Exception as e:
                logger.debug(f"HLS connection failed for {playlist_url}: {e}")
                last_error = e
                if self.cap:
                    self.cap.release()
                    self.cap = None
        
        # All attempts failed
        error_msg = f"Failed to connect to HLS stream. Tried {len(self.playlist_urls)} URLs."
        if last_error:
            error_msg += f" Last error: {last_error}"
        
        logger.error(error_msg)
        logger.error(f"Tried URLs: {', '.join(self.playlist_urls)}")
        self.disconnect()
        return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read frame from HLS stream."""
        if not self._connected or self.cap is None:
            return None
        
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame from HLS stream")
                self._connected = False
                return None
            
            # Validate frame is not corrupted (basic sanity check)
            # ponytail: OpenCV already skips corrupt frames, this catches edge cases
            if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                logger.debug("Received empty/corrupted frame, skipping")
                return None
            
            return frame  # BGR format
            
        except Exception as e:
            logger.error(f"Error reading HLS frame: {e}")
            self._connected = False
            return None
    
    def disconnect(self) -> None:
        """Disconnect from HLS stream."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self._connected = False
        logger.info("HLS stream disconnected")
    
    def get_stream_info(self) -> Dict:
        """Get HLS stream information."""
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

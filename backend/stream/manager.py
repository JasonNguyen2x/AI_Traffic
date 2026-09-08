"""Stream manager for handling video stream lifecycle."""
import threading
import time
import logging
from typing import Optional, Dict
from queue import Queue, Full
import numpy as np

from backend.stream.base import BaseStreamAdapter
from backend.stream.rtsp import RTSPAdapter
from backend.stream.hls import HLSAdapter
from backend.stream.video import VideoAdapter

logger = logging.getLogger(__name__)


class StreamManager:
    """Manages video stream connection and frame buffering."""
    
    def __init__(self, buffer_size: int = 2):
        self.adapter: Optional[BaseStreamAdapter] = None
        self.buffer_size = buffer_size
        self.frame_buffer: Queue = Queue(maxsize=buffer_size)
        
        self.receiver_thread: Optional[threading.Thread] = None
        self.running = False
        
        self.reconnect_delay = 3  # seconds
        self.max_reconnect_attempts = 10
        
        self.lock = threading.Lock()
        
    def connect(self, protocol: str, url: str) -> tuple[bool, str]:
        """
        Connect to a stream using full URL.
        
        Args:
            protocol: Stream protocol ('rtsp', 'hls', or 'video')
            url: Full stream URL or file path
                - RTSP: rtsp://username:password@host:port/path
                - HLS: http://host:port/path/stream.m3u8
                - Video: /path/to/video.mp4
        
        Returns:
            (success: bool, message: str)
        """
        with self.lock:
            # Disconnect existing stream if any
            if self.is_active():
                self.disconnect()
            
            # Create appropriate adapter
            try:
                protocol_lower = protocol.lower()
                
                if protocol_lower == "rtsp":
                    self.adapter = RTSPAdapter(url)
                elif protocol_lower == "hls":
                    self.adapter = HLSAdapter(url)
                elif protocol_lower == "video":
                    self.adapter = VideoAdapter(file_path=url)
                else:
                    return False, f"Unsupported protocol: {protocol}"
                
                # Attempt connection
                logger.info(f"Attempting to connect to {protocol}: {url}")
                
                if not self.adapter.connect():
                    error_msg = f"Cannot connect to {protocol}: {url}"
                    logger.error(error_msg)
                    return False, error_msg
                
                # Verify we can actually get stream info
                info = self.adapter.get_stream_info()
                if not info or info.get('width', 0) == 0:
                    error_msg = f"Connected but no video stream found: {url}"
                    logger.error(error_msg)
                    self.adapter.disconnect()
                    return False, error_msg
                
                # Start receiver thread
                self.running = True
                self.receiver_thread = threading.Thread(
                    target=self._frame_receiver_loop,
                    daemon=True,
                    name="FrameReceiver"
                )
                self.receiver_thread.start()
                
                success_msg = f"Connected: {info['width']}x{info['height']} @ {info['fps']}fps"
                logger.info(success_msg)
                return True, success_msg
                
            except ConnectionRefusedError:
                error_msg = f"Connection refused: Cannot reach {url}"
                logger.error(error_msg)
                return False, error_msg
            except TimeoutError:
                error_msg = f"Connection timeout: {url} not responding"
                logger.error(error_msg)
                return False, error_msg
            except Exception as e:
                error_msg = f"Connection error: {str(e)}"
                logger.error(f"Connection failed: {e}")
                return False, error_msg
    
    def disconnect(self) -> None:
        """Disconnect from stream and stop receiver thread."""
        with self.lock:
            self.running = False
            
            if self.receiver_thread and self.receiver_thread.is_alive():
                self.receiver_thread.join(timeout=2)
            
            if self.adapter:
                self.adapter.disconnect()
                self.adapter = None
            
            # Clear frame buffer
            while not self.frame_buffer.empty():
                try:
                    self.frame_buffer.get_nowait()
                except:
                    break
            
            logger.info("Stream manager disconnected")
    
    def is_active(self) -> bool:
        """Check if stream is active."""
        return self.adapter is not None and self.adapter.is_connected()
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from buffer.
        Returns None if no frame available.
        """
        if self.frame_buffer.empty():
            return None
        
        # Get all frames and return the latest
        latest_frame = None
        while not self.frame_buffer.empty():
            try:
                latest_frame = self.frame_buffer.get_nowait()
            except:
                break
        
        return latest_frame
    
    def get_stream_info(self) -> Optional[Dict]:
        """Get current stream information."""
        if self.adapter:
            return self.adapter.get_stream_info()
        return None
    
    def _frame_receiver_loop(self):
        """Background thread that continuously reads frames."""
        logger.info("Frame receiver thread started")
        reconnect_attempts = 0
        
        # Get video FPS for throttling
        video_fps = 30  # Default
        if self.adapter:
            info = self.adapter.get_stream_info()
            video_fps = info.get('fps', 30) if info else 30
        
        frame_delay = 1.0 / video_fps if video_fps > 0 else 1.0 / 30
        logger.info(f"Frame receiver: Target {video_fps} FPS (delay: {frame_delay:.3f}s)")
        
        while self.running:
            if not self.adapter or not self.adapter.is_connected():
                # Attempt reconnection
                if reconnect_attempts < self.max_reconnect_attempts:
                    reconnect_attempts += 1
                    logger.warning(f"Stream disconnected, reconnecting... (attempt {reconnect_attempts})")
                    time.sleep(self.reconnect_delay)
                    
                    if self.adapter:
                        if self.adapter.connect():
                            logger.info("Reconnection successful")
                            reconnect_attempts = 0
                        else:
                            continue
                else:
                    logger.error("Max reconnection attempts reached, stopping receiver")
                    self.running = False
                    break
            
            frame_start = time.time()
            
            # Read frame
            frame = self.adapter.read_frame()
            
            if frame is not None:
                # Add to buffer (drop oldest if full - latest frame strategy)
                try:
                    self.frame_buffer.put_nowait(frame)
                except Full:
                    # Buffer full, remove oldest and add new
                    try:
                        self.frame_buffer.get_nowait()
                        self.frame_buffer.put_nowait(frame)
                    except:
                        pass
            else:
                # Frame read failed
                logger.warning("Frame read failed")
                time.sleep(0.1)  # Avoid busy loop
                continue
            
            # Throttle to match video FPS
            frame_time = time.time() - frame_start
            sleep_time = max(0, frame_delay - frame_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        logger.info("Frame receiver thread stopped")


# Global singleton instance
_stream_manager: Optional[StreamManager] = None


def get_stream_manager(buffer_size: int = 2) -> StreamManager:
    """Get global stream manager instance."""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = StreamManager(buffer_size=buffer_size)
    return _stream_manager

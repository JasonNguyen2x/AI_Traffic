"""Base stream adapter interface."""
from abc import ABC, abstractmethod
from typing import Optional, Dict
import numpy as np


class BaseStreamAdapter(ABC):
    """Base class for all stream adapters."""
    
    def __init__(self, url: str):
        self.url = url
        self._connected = False
        
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the stream.
        Returns True if successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame from the stream.
        Returns BGR numpy array (H, W, 3) or None if failed.
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the stream and clean up resources."""
        pass
    
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._connected
    
    @abstractmethod
    def get_stream_info(self) -> Dict:
        """
        Get stream information.
        Returns dict with keys: width, height, fps, codec, etc.
        """
        pass
    
    def get_url(self) -> str:
        """Get stream URL."""
        return self.url

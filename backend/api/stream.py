"""Stream API endpoints."""
from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import logging
import tempfile
from pathlib import Path

from backend.stream.manager import get_stream_manager
from backend.core.state import get_state
from backend.core.config import get_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/stream", tags=["stream"])

config = get_config()


class StreamConnectRequest(BaseModel):
    """Stream connection request."""
    protocol: str = Field(..., description="Stream protocol (rtsp, hls, video)")
    url: str = Field(..., description="Full stream URL or file path")


@router.get("/status")
async def get_stream_status():
    """Get current stream status."""
    state = get_state()
    manager = get_stream_manager()
    
    return {
        "active": state.stream_active,
        "status": state.stream_status,
        "config": state.stream_config,
        "info": manager.get_stream_info() if manager.is_active() else None
    }


@router.post("/connect")
async def connect_stream(request: StreamConnectRequest):
    """
    Connect to a video stream.
    
    Examples:
        - RTSP: {"protocol": "rtsp", "url": "rtsp://username:password@192.168.1.100:554/stream1"}
        - HLS: {"protocol": "hls", "url": "http://192.168.1.100:8080/live/stream.m3u8"}
        - Video: {"protocol": "video", "url": "uploads/video.mp4"}
    """
    state = get_state()
    manager = get_stream_manager(buffer_size=config.stream.buffer_size)
    
    # Validate protocol
    if request.protocol.lower() not in config.stream.protocols:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported protocol. Supported: {', '.join(config.stream.protocols)}"
        )
    
    # Check if stream already active
    if manager.is_active():
        # Force disconnect if state is inconsistent
        logger.warning("Stream already active, forcing disconnect")
        manager.disconnect()
        state.stream_active = False
        state.stream_status = "DISCONNECTED"
    
    # Update status
    state.stream_status = "CONNECTING"
    state.stream_config = {
        "protocol": request.protocol,
        "url": request.url,
    }
    
    # Attempt connection
    logger.info(f"Connecting to {request.protocol}: {request.url}")
    
    success, message = manager.connect(
        protocol=request.protocol,
        url=request.url
    )
    
    if success:
        state.stream_active = True
        state.stream_status = "CONNECTED"
        stream_info = manager.get_stream_info()
        
        logger.info(f"Stream connected: {message}")
        
        return {
            "success": True,
            "message": message,
            "config": state.stream_config,
            "info": stream_info
        }
    else:
        state.stream_active = False
        state.stream_status = "ERROR"
        
        logger.error(f"Stream connection failed: {message}")
        
        raise HTTPException(status_code=500, detail=message)


@router.post("/disconnect")
async def disconnect_stream():
    """Disconnect from current stream and cleanup temporary files."""
    state = get_state()
    manager = get_stream_manager()
    
    if not manager.is_active():
        raise HTTPException(status_code=400, detail="No active stream")
    
    logger.info("Disconnecting stream")
    
    # Disconnect
    manager.disconnect()
    
    # Cleanup temporary video file if exists
    if hasattr(state, 'temp_video_path') and state.temp_video_path:
        try:
            temp_path = Path(state.temp_video_path)
            if temp_path.exists():
                temp_path.unlink()
                logger.info(f"Deleted temporary video: {state.temp_video_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temp video: {e}")
        finally:
            state.temp_video_path = None
    
    # Reset state
    state.stream_active = False
    state.stream_status = "DISCONNECTED"
    state.stream_config = None
    state.reset_tracking()
    state.reset_statistics()
    
    return {
        "success": True,
        "message": "Stream disconnected"
    }


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file for testing.
    File is stored in temporary directory and will be auto-deleted on disconnect.
    """
    
    # Validate file type
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Create temporary file (OS will clean up eventually)
        # Use suffix to preserve extension for OpenCV
        temp_file = tempfile.NamedTemporaryFile(
            mode='wb',
            suffix=file_ext,
            delete=False  # We'll delete manually on disconnect
        )
        
        logger.info(f"Uploading video: {file.filename} -> {temp_file.name}")
        
        # Write uploaded content to temp file
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        file_size = len(content)
        logger.info(f"Video uploaded to temp: {temp_file.name} ({file_size} bytes)")
        
        # Store temp file path in state for cleanup
        state = get_state()
        state.temp_video_path = temp_file.name
        
        return {
            "success": True,
            "filename": file.filename,
            "path": temp_file.name,
            "size": file_size
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/frame")
async def get_current_frame():
    """Get current frame as JPEG (for testing)."""
    from backend.stream.manager import get_stream_manager
    import cv2
    from fastapi.responses import Response
    
    manager = get_stream_manager()
    
    if not manager.is_active():
        raise HTTPException(status_code=400, detail="No active stream")
    
    frame = manager.get_latest_frame()
    
    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available")
    
    # Encode frame as JPEG
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to encode frame")
    
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

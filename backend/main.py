"""Main FastAPI application."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from pathlib import Path

from backend.core.config import get_config
from backend.core.state import get_state
from backend.api.stream import router as stream_router

# Load configuration
config = get_config()

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.logging.level),
    format=config.logging.format
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Traffic AI Monitoring System")
    logger.info(f"Server: {config.server.host}:{config.server.port}")
    logger.info(f"AI Device: {config.ai.device}")
    logger.info(f"Supported protocols: {', '.join(config.stream.protocols)}")
    
    # Initialize AI detector
    from backend.ai.vehicle_detector import VehicleDetector
    from backend.ai.plate_detector import PlateDetector
    from backend.ai.helmet_detector import HelmetDetector
    from backend.ai.ocr_reader import PlateOCR
    from pathlib import Path
    
    try:
        logger.info("Loading vehicle detection model...")
        
        # Try TensorRT engine first, fallback to .pt
        vehicle_model = config.models.vehicle
        if not Path(vehicle_model).exists() and hasattr(config.models, 'vehicle_fallback'):
            logger.warning(f"TensorRT engine not found: {vehicle_model}")
            vehicle_model = config.models.vehicle_fallback
            logger.info(f"Using fallback model: {vehicle_model}")
        
        app.state.detector = VehicleDetector(
            model_path=vehicle_model,
            device=config.ai.device,
            confidence=config.ai.vehicle_confidence,
            inference_size=config.ai.inference_size
        )
        
        device_info = app.state.detector.get_device_info()
        logger.info(f"Vehicle detector ready: {device_info}")
        
    except Exception as e:
        logger.error(f"Failed to load vehicle detector: {e}")
        logger.warning("Continuing without AI detection")
        app.state.detector = None
    
    # Initialize plate detector
    try:
        logger.info("Loading license plate detection model...")
        
        # Try TensorRT engine first, fallback to .pt
        plate_model = f"models/{config.models.plate}"
        if not Path(plate_model).exists() and hasattr(config.models, 'plate_fallback'):
            logger.warning(f"TensorRT engine not found: {plate_model}")
            plate_model = f"models/{config.models.plate_fallback}"
            logger.info(f"Using fallback model: {plate_model}")
        
        app.state.plate_detector = PlateDetector(
            model_path=plate_model,
            device=config.ai.device,
            confidence=config.ai.plate_confidence,
            inference_size=config.ai.inference_size
        )
        
        logger.info("Plate detector ready")
        
    except Exception as e:
        logger.error(f"Failed to load plate detector: {e}")
        logger.warning("Continuing without plate detection")
        app.state.plate_detector = None
    
    # Initialize OCR
    try:
        logger.info("Loading OCR reader...")
        app.state.ocr = PlateOCR(
            model_name=config.ocr.model,
            confidence_threshold=config.ocr.confidence_threshold
        )
        
        logger.info("OCR reader ready")
        
    except Exception as e:
        logger.error(f"Failed to load OCR: {e}")
        logger.warning("Continuing without OCR")
        app.state.ocr = None
    
    # Initialize helmet detector
    try:
        logger.info("Loading helmet detection model...")
        
        helmet_model = config.models.helmet
        
        # Try TensorRT engine first, fallback to .pt
        if not Path(helmet_model).exists():
            logger.warning(f"TensorRT engine not found: {helmet_model}")
            helmet_model = config.models.helmet_fallback
            logger.info(f"Using fallback model: {helmet_model}")
        
        if not Path(helmet_model).exists():
            logger.warning(f"Helmet model not found: {helmet_model}")
            app.state.helmet_detector = None
        else:
            app.state.helmet_detector = HelmetDetector(
                model_path=helmet_model,
                device=config.ai.device,
                confidence=config.ai.helmet_confidence,
                inference_size=config.ai.inference_size
            )
            
            logger.info("Helmet detector ready")
        
    except Exception as e:
        logger.error(f"Failed to load helmet detector: {e}")
        logger.warning("Continuing without helmet detection")
        app.state.helmet_detector = None
    
    logger.info("Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    
    # Cleanup
    from backend.stream.manager import get_stream_manager
    manager = get_stream_manager()
    if manager.is_active():
        manager.disconnect()
    
    state = get_state()
    
    # Cleanup temporary video file if exists
    if hasattr(state, 'temp_video_path') and state.temp_video_path:
        try:
            from pathlib import Path
            temp_path = Path(state.temp_video_path)
            if temp_path.exists():
                temp_path.unlink()
                logger.info(f"Deleted temporary video on shutdown: {state.temp_video_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temp video on shutdown: {e}")
    
    state.reset_all()
    
    logger.info("Application stopped")


# Create FastAPI app
app = FastAPI(
    title="Traffic AI Monitoring System",
    description="Real-time vehicle detection and license plate recognition",
    version="1.0.0",
    lifespan=lifespan
)

# Setup static files and templates
from pathlib import Path
import os

# Get base directory
BASE_DIR = Path(__file__).resolve().parent.parent

static_path = BASE_DIR / "static"
templates_path = BASE_DIR / "templates"

if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

templates = Jinja2Templates(directory=str(templates_path))

# Include routers
app.include_router(stream_router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve main page."""
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.get("/api/config")
async def get_api_config():
    """Get system configuration (public parts only)."""
    return {
        "protocols": config.stream.protocols,
        "target_fps": config.video.target_fps,
        "input_resolution": {
            "width": config.video.input_resolution.width,
            "height": config.video.input_resolution.height,
        },
        "output_resolution": {
            "width": config.video.output_resolution.width,
            "height": config.video.output_resolution.height,
        },
    }


@app.get("/api/statistics")
async def get_statistics():
    """Get current traffic statistics."""
    state = get_state()
    return {
        "statistics": state.get_statistics_dict(),
        "recent_plates": state.get_recent_plates_list(limit=10),
        "frame_count": state.frame_count,
    }


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming video and stats."""
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    from backend.stream.manager import get_stream_manager
    from backend.core.state import VehicleState
    import cv2
    import asyncio
    import json
    import time
    from collections import deque
    import concurrent.futures
    
    # P4: Try turbojpeg for 2-3x faster JPEG encoding, fallback to cv2
    turbojpeg_encoder = None
    try:
        from turbojpeg import TurboJPEG, TJPF_BGR
        turbojpeg_encoder = TurboJPEG()
        logger.info("Using TurboJPEG for fast JPEG encoding")
    except (ImportError, OSError, RuntimeError) as e:
        logger.info(f"TurboJPEG not available ({e}), using OpenCV JPEG encoder")
    
    manager = get_stream_manager()
    detector = app.state.detector
    plate_detector = app.state.plate_detector
    helmet_detector = app.state.helmet_detector
    ocr = app.state.ocr
    
    plate_detection_interval = config.ai.plate_detection_interval
    ocr_interval = config.ai.ocr_interval
    helmet_detection_interval = config.ai.helmet_detection_interval
    plate_task = None  # ponytail: single task, avoids buildup
    ocr_task = None
    helmet_task = None
    
    # Reusable thread pool to prevent thread leak
    plate_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="PlateDetection")
    ocr_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="OCR")
    helmet_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="HelmetDetection")
    
    # P3: Cache output dimensions (constant across frames)
    output_width = config.video.output_resolution.width
    output_height = config.video.output_resolution.height
    jpeg_quality = config.video.jpeg_quality
    
    # FPS tracking
    frame_times = deque(maxlen=30)  # Track last 30 frames
    fps_log_interval = 30  # Log FPS every N frames
    fps_log_counter = 0
    
    async def run_plate_detection_async(frame, vehicle_detections, frame_num):
        """Run plate detection in thread pool."""
        loop = asyncio.get_running_loop()
        state = get_state()
        
        def detect_plates():
            # Detect all plates on frame
            all_plates = plate_detector.detect_on_frame(frame)
            logger.info(f"PLATE DETECT: Found {len(all_plates)} plates on frame")
            
            # Associate plates to vehicles (keyed by track_id)
            associations = plate_detector.associate_plates_to_vehicles(all_plates, vehicle_detections)
            logger.info(f"PLATE ASSOC: {len(associations)} vehicles have plates")
            
            return associations
        
        associations = await loop.run_in_executor(plate_executor, detect_plates)
        
        # Cache associations (keyed by track_id)
        state.plate_associations = associations
        state.plate_cache_frame = frame_num
    
    async def run_ocr_async(frame, vehicle_detections, plate_associations, frame_num):
        """Run OCR on detected plates."""
        loop = asyncio.get_running_loop()
        state = get_state()
        
        def read_plates():
            logger.info(f"OCR START: Processing {len(plate_associations)} vehicles with plates")
            results = {}
            crops = {}  # Store vehicle and plate crops for UI
            
            # Build track_id → vehicle lookup
            vehicle_by_id = {}
            for v_idx, v in enumerate(vehicle_detections):
                key = v.get('track_id', v_idx)
                if key == -1:
                    key = v_idx
                vehicle_by_id[key] = v
            
            # Read OCR for each vehicle's plates (keyed by track_id)
            for v_key, plates in plate_associations.items():
                vehicle = vehicle_by_id.get(v_key)
                if vehicle is None:
                    logger.debug(f"OCR SKIP: Track {v_key} not in vehicle_by_id")
                    continue
                
                # Skip OCR if already have confident read
                if v_key in state.tracked_vehicles:
                    vs = state.tracked_vehicles[v_key]
                    if vs.best_plate and len(vs.ocr_history) >= 3:
                        logger.info(f"OCR CACHED: Track {v_key} -> '{vs.best_plate}' (history: {len(vs.ocr_history)})")
                        results[v_key] = vs.best_plate
                        continue
                
                vehicle_bbox = vehicle['bbox']
                vx1, vy1, vx2, vy2 = [int(v) for v in vehicle_bbox]
                
                # Crop vehicle for UI
                vx1_safe = max(0, vx1)
                vy1_safe = max(0, vy1)
                vx2_safe = min(frame.shape[1], vx2)
                vy2_safe = min(frame.shape[0], vy2)
                vehicle_crop = frame[vy1_safe:vy2_safe, vx1_safe:vx2_safe].copy()
                
                # Convert plate bbox to frame coords
                if plates:
                    plate = plates[0]
                    rel_x1, rel_y1, rel_x2, rel_y2 = plate['bbox_rel']
                    
                    v_width = vx2 - vx1
                    v_height = vy2 - vy1
                    
                    px1 = vx1 + rel_x1 * v_width
                    py1 = vy1 + rel_y1 * v_height
                    px2 = vx1 + rel_x2 * v_width
                    py2 = vy1 + rel_y2 * v_height
                    
                    plate_bbox = [px1, py1, px2, py2]
                    
                    # Crop plate for UI
                    px1_safe = max(0, int(px1))
                    py1_safe = max(0, int(py1))
                    px2_safe = min(frame.shape[1], int(px2))
                    py2_safe = min(frame.shape[0], int(py2))
                    plate_crop = frame[py1_safe:py2_safe, px1_safe:px2_safe].copy()
                    
                    logger.info(f"OCR ATTEMPT: Track {v_key}, bbox [{px1:.0f},{py1:.0f},{px2:.0f},{py2:.0f}]")
                    
                    # Run fast OCR
                    text = ocr.read_plate(frame, plate_bbox)
                    
                    if text:
                        logger.info(f"OCR RESULT: Track {v_key} -> '{text}' (len={len(text)}, bbox: [{px1:.0f},{py1:.0f},{px2:.0f},{py2:.0f}])")
                        results[v_key] = text
                        
                        # ponytail: Only create crops if text >= 8 chars (Vietnamese plate format)
                        if len(text) >= 8:
                            # Store crops for UI
                            crops[v_key] = {
                                'vehicle_crop': vehicle_crop,
                                'plate_crop': plate_crop,
                                'text': text,
                                'track_id': v_key
                            }
                            logger.info(f"OCR CROP SAVED: Track {v_key} -> '{text}' (valid length)")
                        else:
                            logger.info(f"OCR CROP SKIPPED: Track {v_key} -> '{text}' (too short, need >= 8 chars)")
                        
                        # Temporal voting
                        if v_key in state.tracked_vehicles:
                            state.tracked_vehicles[v_key].add_ocr_result(
                                text, plate['confidence']
                            )
                            best = state.tracked_vehicles[v_key].best_plate
                            if best and best != text:
                                logger.info(f"OCR VOTED: Track {v_key} -> '{best}' (after temporal voting)")
                                results[v_key] = best
                                # Update crop text if already saved
                                if v_key in crops:
                                    crops[v_key]['text'] = best
                    else:
                        logger.info(f"OCR NO TEXT: Track {v_key} (bbox: [{px1:.0f},{py1:.0f},{px2:.0f},{py2:.0f}])")
                else:
                    logger.info(f"OCR SKIP: Track {v_key} has no plates in association")
            
            return results, crops
        
        results, crops = await loop.run_in_executor(ocr_executor, read_plates)
        
        # Cache OCR results
        state.ocr_results = results
        state.ocr_cache_frame = frame_num
        
        return crops  # Return crops for WebSocket transmission
    
    async def run_helmet_detection_async(frame, vehicle_detections, frame_num):
        """Run helmet detection in thread pool."""
        loop = asyncio.get_running_loop()
        state = get_state()
        
        def detect_helmets():
            # Detect all helmets on frame
            all_helmets = helmet_detector.detect_on_frame(frame)
            
            # Associate helmets to motorcycles (keyed by vehicle_idx)
            associations = helmet_detector.associate_helmets_to_motorcycles(all_helmets, vehicle_detections)
            
            return associations
        
        associations = await loop.run_in_executor(helmet_executor, detect_helmets)
        
        # Cache associations and update VehicleState
        state.helmet_associations = associations
        state.helmet_cache_frame = frame_num
        
        # Update tracked vehicles with helmet status
        for v_idx, helmet_info in associations.items():
            if v_idx < len(vehicle_detections):
                vehicle = vehicle_detections[v_idx]
                tid = vehicle.get('track_id', -1)
                
                if tid != -1 and tid in state.tracked_vehicles:
                    state.tracked_vehicles[tid].has_helmet = helmet_info['has_helmet']
                    state.tracked_vehicles[tid].helmet_confidence = helmet_info['confidence']
    
    try:
        stream_info = None
        frame_delay = 1.0 / 30
        
        while True:
            if not manager.is_active():
                await asyncio.sleep(0.5)
                continue
            
            if stream_info is None:
                stream_info = manager.get_stream_info()
                if stream_info:
                    fps = stream_info.get('fps', 30)
                    frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30
                    logger.info(f"WebSocket: Streaming at {fps} FPS (delay: {frame_delay:.3f}s)")
            
            frame_start = time.time()
            frame = manager.get_latest_frame()
            
            if frame is not None:
                state = get_state()
                
                # === TIMING: Start ===
                t_total_start = time.perf_counter()
                t_inference = 0
                t_draw = 0
                t_resize = 0
                t_encode = 0
                
                # Vehicle detection + tracking
                detections = []
                if detector is not None:
                    t0 = time.perf_counter()
                    detections = detector.track(frame)  # ByteTrack: returns track_id per detection
                    t_inference = (time.perf_counter() - t0) * 1000
                    
                    # Log every 30 frames for debug
                    if state.frame_count % 30 == 0:
                        logger.info(f"FRAME {state.frame_count}: Detected {len(detections)} vehicles")
                    
                    # Update VehicleState for each tracked detection
                    for det in detections:
                        tid = det.get('track_id', -1)
                        if tid == -1:
                            continue
                        if tid not in state.tracked_vehicles:
                            state.tracked_vehicles[tid] = VehicleState(
                                track_id=tid,
                                vehicle_class=det['class'],
                                last_seen_frame=state.frame_count
                            )
                        else:
                            state.tracked_vehicles[tid].last_seen_frame = state.frame_count
                    
                    if len(detections) > 0:
                        # Trigger plate detection async (every N frames)
                        if plate_detector is not None and state.frame_count % plate_detection_interval == 0:
                            # Cancel and await previous task properly
                            if plate_task is not None and not plate_task.done():
                                plate_task.cancel()
                                try:
                                    await plate_task
                                except asyncio.CancelledError:
                                    pass
                            
                            # No frame.copy() - pass original frame (executor will work on it immediately)
                            plate_task = asyncio.create_task(
                                run_plate_detection_async(frame, detections, state.frame_count)
                            )
                        
                        # Trigger OCR async (every M frames, after plates detected)
                        if ocr is not None and state.plate_associations and state.frame_count % ocr_interval == 0:
                            if ocr_task is None or ocr_task.done():
                                logger.info(f"Triggering OCR at frame {state.frame_count}")
                                ocr_task = asyncio.create_task(
                                    run_ocr_async(frame, detections, state.plate_associations, state.frame_count)
                                )
                            else:
                                logger.debug(f"OCR skipped - previous task still running")
                        elif ocr is not None and state.plate_associations:
                            logger.debug(f"OCR waiting for interval (frame {state.frame_count} % {ocr_interval} != 0)")
                        
                        # Trigger helmet detection async (every K frames, independent of vehicle class)
                        # ponytail: Helmet detector runs in parallel, association filters motorcycles later
                        if helmet_detector is not None and state.frame_count % helmet_detection_interval == 0:
                            # Cancel and await previous task properly
                            if helmet_task is not None and not helmet_task.done():
                                helmet_task.cancel()
                                try:
                                    await helmet_task
                                except asyncio.CancelledError:
                                    pass
                            
                            # No frame.copy() - pass original frame
                            helmet_task = asyncio.create_task(
                                run_helmet_detection_async(frame, detections, state.frame_count)
                            )
                        
                        # Draw vehicles
                        t0 = time.perf_counter()
                        frame = detector.draw_detections(frame, detections)
                        
                        # Draw cached plates with OCR text
                        if plate_detector is not None and state.plate_associations:
                            frame = plate_detector.draw_plates_from_associations(
                                frame, detections, state.plate_associations, state.ocr_results
                            )
                        
                        # Draw helmet status on motorcycles
                        if helmet_detector is not None and state.helmet_associations:
                            frame = helmet_detector.draw_helmets_from_associations(
                                frame, detections, state.helmet_associations
                            )
                        
                        t_draw = (time.perf_counter() - t0) * 1000
                
                # Update density
                vehicle_count = len(detections)
                state.statistics.current_vehicles = vehicle_count
                
                # Cleanup stale tracked vehicles periodically
                if state.frame_count % 100 == 0:
                    state.cleanup_old_vehicles(state.frame_count, max_age=100)
                    # Also cleanup plate/OCR/helmet caches if stale
                    if state.frame_count - state.plate_cache_frame > 50:
                        state.plate_associations.clear()
                    if state.frame_count - state.ocr_cache_frame > 50:
                        state.ocr_results.clear()
                    if state.frame_count - state.helmet_cache_frame > 50:
                        state.helmet_associations.clear()
                
                if vehicle_count < config.density.low:
                    density_level = "low"
                elif vehicle_count <= config.density.medium:
                    density_level = "medium"
                else:
                    density_level = "high"
                
                state.statistics.density_level = density_level
                
                # P3: Skip resize when frame dimensions already match
                t0 = time.perf_counter()
                h, w = frame.shape[:2]
                if w != output_width or h != output_height:
                    frame_out = cv2.resize(frame, (output_width, output_height))
                else:
                    frame_out = frame
                t_resize = (time.perf_counter() - t0) * 1000
                
                # P4: Use TurboJPEG if available (2-3x faster), fallback to cv2
                t0 = time.perf_counter()
                if turbojpeg_encoder is not None:
                    buffer = turbojpeg_encoder.encode(frame_out, quality=jpeg_quality)
                    ret = buffer is not None
                else:
                    ret, buffer = cv2.imencode('.jpg', frame_out, 
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                    if ret:
                        buffer = buffer.tobytes()
                t_encode = (time.perf_counter() - t0) * 1000
                
                # === TIMING: End ===
                t_total = (time.perf_counter() - t_total_start) * 1000
                
                # Track FPS
                frame_times.append(t_total)
                fps_log_counter += 1
                
                # Calculate and log FPS every N frames
                if fps_log_counter >= fps_log_interval:
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    current_fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
                    
                    logger.info(
                        f"FPS: {current_fps:.1f} | "
                        f"Frame: {avg_frame_time:.1f}ms | "
                        f"Inference: {t_inference:.1f}ms | "
                        f"Draw: {t_draw:.1f}ms | "
                        f"Resize: {t_resize:.1f}ms | "
                        f"Encode: {t_encode:.1f}ms | "
                        f"Vehicles: {vehicle_count}"
                    )
                    fps_log_counter = 0
                
                if ret:
                    await websocket.send_bytes(buffer if isinstance(buffer, bytes) else buffer.tobytes())
                
                state.frame_count += 1
                
                # Send stats every frame
                stats_msg = {
                    "type": "stats",
                    "total": state.statistics.total,
                    "cars": state.statistics.cars,
                    "motorcycles": state.statistics.motorcycles,
                    "trucks": state.statistics.trucks,
                    "buses": state.statistics.buses,
                    "in": state.statistics.in_count,
                    "out": state.statistics.out_count,
                    "current_vehicles": state.statistics.current_vehicles,
                    "density_level": density_level,
                    "fps": round(1000 / t_total, 1) if t_total > 0 else 0,
                    "frame_time": round(t_total, 1)
                }
                await websocket.send_text(json.dumps(stats_msg))
                
                # Check if OCR task completed and send plate crops
                if ocr_task is not None and ocr_task.done():
                    try:
                        crops = await ocr_task
                        logger.info(f"OCR task completed, got {len(crops) if crops else 0} crops")
                        if crops:
                            # Encode and send each plate detection
                            for track_id, crop_data in crops.items():
                                vehicle_crop = crop_data['vehicle_crop']
                                plate_crop = crop_data['plate_crop']
                                
                                logger.info(f"Sending plate_detection for track {track_id}: text='{crop_data['text']}', v_shape={vehicle_crop.shape}, p_shape={plate_crop.shape}")
                                
                                # Encode crops to JPEG base64
                                _, v_buffer = cv2.imencode('.jpg', vehicle_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                _, p_buffer = cv2.imencode('.jpg', plate_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                
                                import base64
                                vehicle_b64 = base64.b64encode(v_buffer).decode('utf-8')
                                plate_b64 = base64.b64encode(p_buffer).decode('utf-8')
                                
                                plate_msg = {
                                    "type": "plate_detection",
                                    "track_id": track_id,
                                    "text": crop_data['text'],
                                    "vehicle_image": vehicle_b64,
                                    "plate_image": plate_b64,
                                    "timestamp": time.time()
                                }
                                await websocket.send_text(json.dumps(plate_msg))
                                logger.info(f"Sent plate_detection message for track {track_id}")
                        else:
                            logger.info("OCR task returned no crops")
                        
                        ocr_task = None  # Clear task
                    except Exception as e:
                        logger.error(f"Failed to send plate crops: {e}", exc_info=True)
            
            frame_time = time.time() - frame_start
            sleep_time = max(0, frame_delay - frame_time)
            await asyncio.sleep(sleep_time)
                    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except asyncio.CancelledError:
        logger.info("WebSocket task cancelled (server shutdown)")
        raise
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Properly cleanup tasks
        if plate_task is not None and not plate_task.done():
            plate_task.cancel()
            try:
                await plate_task
            except asyncio.CancelledError:
                pass
        if ocr_task is not None and not ocr_task.done():
            ocr_task.cancel()
            try:
                await ocr_task
            except asyncio.CancelledError:
                pass
        if helmet_task is not None and not helmet_task.done():
            helmet_task.cancel()
            try:
                await helmet_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown thread pools
        plate_executor.shutdown(wait=False)
        ocr_executor.shutdown(wait=False)
        helmet_executor.shutdown(wait=False)
        
        # Reset tracker state for next connection
        if detector is not None:
            detector.reset_tracker()
        
        logger.info("WebSocket connection closed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload
    )

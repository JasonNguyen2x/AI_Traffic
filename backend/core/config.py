"""Configuration loader for the application."""
import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


class AIConfig(BaseModel):
    device: str = "auto"
    vehicle_confidence: float = 0.4
    plate_confidence: float = 0.4
    helmet_confidence: float = 0.5
    inference_size: int = 640
    plate_detection_interval: int = 5
    ocr_interval: int = 10
    helmet_detection_interval: int = 3


class ModelsConfig(BaseModel):
    vehicle: str = "models/yolov8n.pt"
    plate: str = "models/license_plate.pt"
    helmet: str = "models/helmet_detector.pt"


class TrackingConfig(BaseModel):
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3


class StreamConfig(BaseModel):
    buffer_size: int = 2
    reconnect_delay: int = 3
    max_reconnect_attempts: int = 10
    connection_timeout: int = 5
    frame_timeout: int = 10
    protocols: list[str] = ["rtsp", "hls"]


class ResolutionConfig(BaseModel):
    width: int
    height: int


class VideoConfig(BaseModel):
    input_resolution: ResolutionConfig
    output_resolution: ResolutionConfig
    target_fps: int = 30
    jpeg_quality: int = 85


class LineConfig(BaseModel):
    x1: int = 100
    y1: int = 300
    x2: int = 900
    y2: int = 300


class CountingConfig(BaseModel):
    line: LineConfig
    direction: str = "both"


class MemoryConfig(BaseModel):
    max_tracked_vehicles: int = 1000
    max_ocr_history_per_vehicle: int = 20
    max_recent_plates: int = 100
    cleanup_interval_frames: int = 100


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "[%(levelname)s] %(asctime)s - %(message)s"


class DensityConfig(BaseModel):
    low: int = 5
    medium: int = 10


class OCRConfig(BaseModel):
    model: str = "cct-xs-v2-global-model"
    confidence_threshold: float = 0.6
    temporal_voting_frames: int = 5


class Config(BaseModel):
    server: ServerConfig
    ai: AIConfig
    models: ModelsConfig
    tracking: TrackingConfig
    stream: StreamConfig
    video: VideoConfig
    counting: CountingConfig
    ocr: OCRConfig
    memory: MemoryConfig
    logging: LoggingConfig
    density: DensityConfig


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(**config_dict)


# Global config instance
config: Config | None = None


def get_config() -> Config:
    """Get global config instance."""
    global config
    if config is None:
        config = load_config()
    return config

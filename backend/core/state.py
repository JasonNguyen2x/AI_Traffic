"""Application state management (in-memory only)."""
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class VehicleState:
    """State for a tracked vehicle."""
    track_id: int
    vehicle_class: str
    last_seen_frame: int
    ocr_history: List[str] = field(default_factory=list)
    ocr_confidences: List[float] = field(default_factory=list)
    best_plate: Optional[str] = None
    crossed_line: bool = False
    counted: bool = False
    first_seen: datetime = field(default_factory=datetime.now)
    # Helmet status (for motorcycles)
    has_helmet: Optional[bool] = None
    helmet_confidence: float = 0.0
    
    def add_ocr_result(self, text: str, confidence: float, max_history: int = 20):
        """Add OCR result with temporal voting."""
        self.ocr_history.append(text)
        self.ocr_confidences.append(confidence)
        
        # Limit history size
        if len(self.ocr_history) > max_history:
            self.ocr_history.pop(0)
            self.ocr_confidences.pop(0)
        
        # Update best plate using temporal voting
        self.best_plate = self._compute_best_plate()
    
    def _compute_best_plate(self) -> Optional[str]:
        """Compute best plate using temporal voting."""
        if not self.ocr_history:
            return None
        
        # Count occurrences with confidence weighting
        from collections import defaultdict
        weighted_votes = defaultdict(float)
        
        for text, conf in zip(self.ocr_history, self.ocr_confidences):
            weighted_votes[text] += conf
        
        # Return plate with highest weighted votes
        if weighted_votes:
            best = max(weighted_votes.items(), key=lambda x: x[1])
            return best[0]
        
        return None


@dataclass
class Statistics:
    """Traffic statistics."""
    total: int = 0
    cars: int = 0
    motorcycles: int = 0
    trucks: int = 0
    buses: int = 0
    in_count: int = 0
    out_count: int = 0
    current_vehicles: int = 0  # Current vehicles in frame
    density_level: str = "low"  # low, medium, high


@dataclass
class PlateDetection:
    """License plate detection result."""
    track_id: int
    plate_number: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


class ApplicationState:
    """Global application state."""
    
    def __init__(self):
        self.stream_active: bool = False
        self.stream_status: str = "DISCONNECTED"
        self.stream_config: Optional[Dict] = None
        
        self.statistics = Statistics()
        self.tracked_vehicles: Dict[int, VehicleState] = {}
        self.counted_ids: Set[int] = set()
        self.recent_plates: List[PlateDetection] = []
        
        self.frame_count: int = 0
        
        # Plate detection cache (track_id -> list of plates)
        self.plate_associations: Dict[int, List[Dict]] = {}
        self.plate_cache_frame: int = -1
        
        # OCR results cache (track_id -> plate_text)
        self.ocr_results: Dict[int, str] = {}
        self.ocr_cache_frame: int = -1
        
        # Helmet detection cache (vehicle_idx -> helmet_info)
        self.helmet_associations: Dict[int, Dict] = {}
        self.helmet_cache_frame: int = -1
        
        # Temporary video file path (for cleanup on disconnect)
        self.temp_video_path: Optional[str] = None
        
    def reset_statistics(self):
        """Reset all statistics."""
        self.statistics = Statistics()
        self.counted_ids.clear()
        
    def reset_tracking(self):
        """Reset tracking state."""
        self.tracked_vehicles.clear()
        self.counted_ids.clear()
        
    def reset_all(self):
        """Reset everything."""
        self.stream_active = False
        self.stream_status = "DISCONNECTED"
        self.stream_config = None
        self.reset_statistics()
        self.reset_tracking()
        self.recent_plates.clear()
        self.frame_count = 0
        self.plate_associations.clear()
        self.plate_cache_frame = -1
        self.ocr_results.clear()
        self.ocr_cache_frame = -1
        self.helmet_associations.clear()
        self.helmet_cache_frame = -1
        self.temp_video_path = None
        
    def add_plate_detection(self, detection: PlateDetection, max_recent: int = 100):
        """Add a plate detection to recent list."""
        self.recent_plates.insert(0, detection)
        if len(self.recent_plates) > max_recent:
            self.recent_plates = self.recent_plates[:max_recent]
            
    def cleanup_old_vehicles(self, current_frame: int, max_age: int = 100):
        """Remove vehicles not seen recently."""
        to_remove = [
            track_id for track_id, vehicle in self.tracked_vehicles.items()
            if current_frame - vehicle.last_seen_frame > max_age
        ]
        for track_id in to_remove:
            del self.tracked_vehicles[track_id]
            
    def get_statistics_dict(self) -> Dict:
        """Get statistics as dictionary."""
        return {
            "total": self.statistics.total,
            "cars": self.statistics.cars,
            "motorcycles": self.statistics.motorcycles,
            "trucks": self.statistics.trucks,
            "buses": self.statistics.buses,
            "in": self.statistics.in_count,
            "out": self.statistics.out_count,
            "current_vehicles": self.statistics.current_vehicles,
            "density_level": self.statistics.density_level,
        }
        
    def get_recent_plates_list(self, limit: int = 10) -> List[Dict]:
        """Get recent plates as list of dicts."""
        return [
            {
                "track_id": p.track_id,
                "plate_number": p.plate_number,
                "confidence": round(p.confidence, 2),
                "timestamp": p.timestamp.strftime("%H:%M:%S"),
            }
            for p in self.recent_plates[:limit]
        ]


# Global state instance
app_state = ApplicationState()


def get_state() -> ApplicationState:
    """Get global application state."""
    return app_state

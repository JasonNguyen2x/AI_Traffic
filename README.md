# AI Traffic Monitoring System

Real-time traffic monitoring system with AI-powered vehicle detection, license plate recognition (OCR), and motorcycle helmet detection.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)

---

## Features

### Vehicle Detection & Tracking
- Real-time detection of cars, motorcycles, trucks, and buses
- YOLO11n-based object detection
- ByteTrack multi-object tracking with stable IDs
- TensorRT acceleration support (3-5x faster inference)

### License Plate Recognition
- Automatic license plate detection
- Fast OCR using **fast-plate-ocr** (30-50x faster than EasyOCR)
- ONNX GPU-optimized inference
- Temporal voting for stable OCR results (>=8 chars Vietnamese plates)
- Vehicle and plate image crops for verification

### Helmet Detection (NEW!)
- Real-time helmet detection for motorcycles
- Spatial association using BikeWithRider → Helmet/NoHelmet logic
- Color-coded visualization:
  - **Yellow box**: Helmet detected
  - **Red box**: No helmet detected
- Configurable detection intervals for performance tuning

### Traffic Analysis
- Traffic density monitoring (Low/Medium/High)
- Virtual counting line for IN/OUT statistics
- Real-time FPS and performance metrics
- Vehicle class breakdown

### Streaming & UI
- Multiple protocol support: **RTSP**, **HLS**, **Video Files**
- Real-time WebSocket streaming (~30 FPS)
- Modern responsive web interface
- Live statistics dashboard
- License plate display panel

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     VIDEO STREAM (30 FPS)                   │
│                   RTSP / HLS / Video File                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │   Stream Manager       │
            │   - RTSP Adapter       │
            │   - HLS Adapter        │
            │   - Video File Adapter │
            └────────────┬───────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  Vehicle Detection     │
            │  YOLO11n + ByteTrack   │
            │  (TensorRT optimized)  │
            └────────┬───────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ Plate   │  │ Helmet  │  │ Traffic │
  │ Detect  │  │ Detect  │  │ Counting│
  │ (every  │  │ (every  │  │ (virtual│
  │ 5 frame)│  │ 3 frame)│  │  line)  │
  └────┬────┘  └────┬────┘  └────┬────┘
       │            │            │
       ▼            │            │
  ┌─────────┐      │            │
  │   OCR   │      │            │
  │fast-plate│     │            │
  │  (every  │     │            │
  │ 5 frame) │     │            │
  └────┬─────┘     │            │
       │           │            │
       └───────────┴────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   Frame Rendering    │
        │   - Vehicle boxes    │
        │   - Plate boxes      │
        │   - Helmet boxes     │
        │   - Count line       │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   WebSocket Stream   │
        │   JPEG encoding      │
        │   → Browser Display  │
        └──────────────────────┘
```

---

## Quick Start

### 1. Prerequisites

- **Python**: 3.11 or higher
- **GPU** (recommended): NVIDIA GPU with CUDA 13.0 support (RTX 5050 8GB or better)
- **RAM**: 16GB recommended
- **Storage**: 10GB for models and dependencies

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AI_Traffic.git
cd AI_Traffic

# Install dependencies
pip install -r requirements.txt
```

**For CUDA 13.0 support** (RTX 5050/5060 series):
```bash
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
```

### 3. Download Models

Place these models in the `models/` directory:

| Model | Purpose | Required |
|-------|---------|----------|
| `yolo11n.pt` / `.engine` | Vehicle detection | Yes |
| `license_plate_detector.pt` / `.engine` | Plate detection | Yes |
| `helmet_detector.pt` / `.engine` | Helmet detection | Optional |

**Download YOLO11n automatically**:
```python
from ultralytics import YOLO
model = YOLO('yolo11n.pt')  # Auto-downloads
```

**TensorRT engines** (for 3-5x speedup):
```bash
python setup_tensorrt.py
```

### 4. Configure

Edit `config.yaml` to customize settings:

```yaml
# Server
server:
  host: 0.0.0.0
  port: 8500

# AI Parameters
ai:
  device: auto  # auto, cuda, or cpu
  vehicle_confidence: 0.25
  plate_confidence: 0.35
  helmet_confidence: 0.5
  inference_size: 1280
  plate_detection_interval: 5
  helmet_detection_interval: 3

# Models (TensorRT engines or .pt fallback)
models:
  vehicle: models/yolo11n.engine
  vehicle_fallback: models/yolo11n.pt
  plate: license_plate_detector.engine
  plate_fallback: license_plate_detector.pt
  helmet: models/helmet_detector.engine
  helmet_fallback: models/helmet_detector.pt

# OCR (fast-plate-ocr)
ocr:
  model: cct-xs-v2-global-model  # Fast & accurate
  confidence_threshold: 0.0  # Disabled (char_probs used instead)
  temporal_voting_frames: 5
```

### 5. Start Server

```bash
# Method 1: Direct Python
python -m backend.main

# Method 2: Uvicorn
uvicorn backend.main:app --host 0.0.0.0 --port 8500

# Method 3: With hot reload (development)
uvicorn backend.main:app --host 0.0.0.0 --port 8500 --reload
```

### 6. Access Web Interface

Open browser: **http://localhost:8500**

---

## Usage

### Upload Video File

1. Click **"Settings"** button (or press `G`)
2. Select **"Video File (Test)"**
3. Choose a video file (MP4, AVI, MOV)
4. Click **"Connect"**

### Connect to RTSP Stream

1. Protocol: **RTSP**
2. Host: `192.168.1.100` (camera IP)
3. Port: `554`
4. Path: `/stream1` (optional)
5. Click **"Connect"**

**Example RTSP URL**: `rtsp://192.168.1.100:554/stream1`

### Connect to HLS Stream

1. Protocol: **HLS**
2. Host: `192.168.1.100`
3. Port: `8080`
4. Path: `/live/stream.m3u8`
5. Click **"Connect"**

**Example HLS URL**: `http://192.168.1.100:8080/live/stream.m3u8`

---

## Visual Output

### Vehicle Detection
```
┌─────────────────────────────────────┐
│  Car #5          Motorcycle #12     │
│  [green box]        [green box]     │
│  conf: 0.87         conf: 0.92      │
│                                     │
│  Truck #8        Bus #3             │
│  [green box]        [green box]     │
│  conf: 0.91         conf: 0.88      │
└─────────────────────────────────────┘
```

### License Plate Detection
```
┌──────────────────────┐
│  Vehicle #5          │
│  ┌──────────┐        │
│  │ 30L21050 │ ← Plate box (blue)
│  └──────────┘        │
│  OCR: 30L21050       │
│  Conf: 0.97          │
└──────────────────────┘

UI Sidebar:
┌─────────────────────────┐
│ LICENSE PLATES          │
├─────────────────────────┤
│ Vehicle #5              │
│ [vehicle_crop] [plate]  │
│ 30L21050                │
├─────────────────────────┤
│ Vehicle #12             │
│ [vehicle_crop] [plate]  │
│ 29H18115                │
└─────────────────────────┘
```

### Helmet Detection (NEW!)
```
With helmet
┌──────────────────────┐
│  Motorcycle #12      │
│  [yellow helmet box] │
│  Helmet 0.75         │
└──────────────────────┘

Without helmet
┌──────────────────────┐
│  Motorcycle #15      │
│  [red nohelmet box]  │
│  NoHelmet 0.82       │
└──────────────────────┘
```

### Traffic Counting
```
────────────────────────────────
   Counting Line (cyan)
────────────────────────────────

IN: ↓ 45 vehicles
OUT: ↑ 38 vehicles
```

---

## Configuration Guide

### Performance Tuning

| Goal | Vehicle Conf | Plate Conf | Helmet Conf | Intervals |
|------|--------------|------------|-------------|-----------|
| **High Accuracy** | 0.4 | 0.4 | 0.7 | plate:10, helmet:5 |
| **Balanced** | 0.25 | 0.35 | 0.5 | plate:5, helmet:3 ⭐ |
| **Performance** | 0.2 | 0.3 | 0.4 | plate:10, helmet:10 |

### Resolution Tuning

```yaml
video:
  input_resolution:
    width: 1280   # Model inference size
    height: 720
  output_resolution:
    width: 1280   # WebSocket stream size
    height: 720
  jpeg_quality: 75  # 0-100 (higher = better quality, larger size)
```

**Lower resolution** → Faster FPS, less accurate  
**Higher resolution** → Slower FPS, more accurate

### TensorRT Acceleration

**Before** (PyTorch .pt):
- Vehicle detection: ~30-40ms
- Plate detection: ~25-30ms
- Total: ~60ms → **~16 FPS**

**After** (TensorRT .engine):
- Vehicle detection: ~8-10ms
- Plate detection: ~6-8ms
- Total: ~20ms → **~50 FPS**

**Build engines**:
```bash
python setup_tensorrt.py
```

---

## Project Structure

```
AI_Traffic/
├── backend/
│   ├── main.py                    # FastAPI application
│   ├── api/
│   │   └── stream.py              # REST API endpoints
│   ├── ai/
│   │   ├── vehicle_detector.py    # YOLO vehicle detection
│   │   ├── plate_detector.py      # License plate detection
│   │   ├── ocr_reader.py          # fast-plate-ocr integration
│   │   └── helmet_detector.py     # Helmet detection (NEW!)
│   ├── stream/
│   │   ├── base.py                # Base stream adapter
│   │   ├── rtsp.py                # RTSP adapter
│   │   ├── hls.py                 # HLS adapter
│   │   ├── video.py               # Video file adapter
│   │   └── manager.py             # Stream manager
│   └── core/
│       ├── config.py              # Configuration loader
│       └── state.py               # Application state
├── templates/
│   └── index.html                 # Web interface
├── static/
│   ├── css/style.css              # Styles
│   └── js/app.js                  # Frontend logic
├── models/                        # AI models
│   ├── yolo11n.pt / .engine
│   ├── license_plate_detector.pt / .engine
│   └── helmet_detector.pt / .engine
├── docs/                          # Documentation
│   ├── HELMET_DETECTION.md
│   ├── HELMET_FEATURE_SUMMARY.md
│   ├── PHASE_*.md
│   └── TENSORRT_SETUP.md
├── uploads/                       # Uploaded video files
├── config.yaml                    # System configuration
├── requirements.txt               # Python dependencies
├── setup_tensorrt.py              # TensorRT engine builder
├── INSTALLATION_GUIDE.md          # Detailed installation
├── HELMET_QUICKSTART.md           # Helmet detection guide
└── README.md                      # This file
```

---

## Technology Stack

### Backend
- **FastAPI** - Modern async web framework
- **Uvicorn** - ASGI server
- **WebSocket** - Real-time bidirectional communication
- **OpenCV** - Video processing
- **FFmpeg** - Stream decoding (RTSP/HLS)

### AI/ML
- **PyTorch 2.12** - Deep learning framework
- **Ultralytics YOLO11** - Object detection
- **ByteTrack** - Multi-object tracking
- **fast-plate-ocr** - License plate OCR (ONNX GPU-optimized)
- **TensorRT 10.0+** - Inference acceleration (RTX 5000 series)

### Frontend
- **Vanilla JavaScript** - No framework overhead
- **HTML5 Canvas / Image** - Video rendering
- **CSS Grid/Flexbox** - Responsive layout
- **WebSocket API** - Real-time updates

---

## Performance Metrics

### Hardware: NVIDIA RTX 5050 8GB, 1280x720 resolution

| Component | Inference Time | Interval | Avg Impact |
|-----------|----------------|----------|------------|
| Vehicle Detection | ~8-10ms (TensorRT) | Every frame | 10ms |
| Plate Detection | ~6-8ms (TensorRT) | Every 5 frames | ~2ms |
| Helmet Detection | ~6-8ms (TensorRT) | Every 3 frames | ~3ms |
| OCR | ~2-3ms (ONNX GPU) | Every 5 frames | ~0.5ms |
| Frame Rendering | ~5-8ms | Every frame | 7ms |
| **Total Average** | - | - | **~22ms → 45 FPS** |

### FPS by Vehicle Count

| Vehicles | Motorcycles | FPS | Notes |
|----------|-------------|-----|-------|
| 0-5 | 0-2 | 40-45 | Minimal overhead |
| 5-10 | 2-5 | 35-40 | Smooth |
| 10-20 | 5-10 | 28-35 | Still good |
| 20+ | 10+ | 20-28 | May need tuning |

---

## API Endpoints

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web interface |
| `GET` | `/api/config` | System configuration |
| `GET` | `/api/stream/status` | Stream connection status |
| `GET` | `/api/statistics` | Traffic statistics |
| `POST` | `/api/stream/connect` | Connect to stream |
| `POST` | `/api/stream/disconnect` | Disconnect stream |
| `POST` | `/api/upload` | Upload video file |

### WebSocket

**Endpoint**: `/ws/stream`

**Client → Server** (JSON):
```json
{
  "action": "start_stream"
}
```

**Server → Client** (Binary):
```
JPEG frame data (30 FPS)
```

**Server → Client** (JSON):
```json
{
  "type": "stats",
  "total": 12,
  "cars": 5,
  "motorcycles": 4,
  "trucks": 2,
  "buses": 1,
  "in": 45,
  "out": 38,
  "density": "medium",
  "fps": 28.5
}
```

```json
{
  "type": "plate_detection",
  "track_id": 5,
  "vehicle_crop": "data:image/jpeg;base64,...",
  "plate_crop": "data:image/jpeg;base64,...",
  "text": "30L21050"
}
```

---

## Testing

### Unit Tests

```bash
# Test helmet detection
python test_helmet_detection.py

# Test all components
test_all.bat  # Windows
./test_all.sh  # Linux/Mac
```

### Manual Testing

1. **Vehicle Detection**: Upload video with traffic
2. **Plate Detection**: Use video with visible license plates
3. **Helmet Detection**: Use video with motorcycles
4. **RTSP Stream**: Connect to IP camera
5. **HLS Stream**: Connect to HLS server

---

## Troubleshooting

### CUDA Not Available

**Symptom**: `CUDA requested but not available, falling back to CPU`

**Check**:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Solutions**:
1. Install/update NVIDIA drivers
2. Install CUDA Toolkit 13.0
3. Reinstall PyTorch with CUDA:
   ```bash
   pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
   ```

### Low FPS / Performance Issues

**Solutions**:
1. **Enable TensorRT**: Run `python setup_tensorrt.py`
2. **Increase detection intervals**: Edit `config.yaml`
   ```yaml
   plate_detection_interval: 10  # From 5
   helmet_detection_interval: 10  # From 3
   ```
3. **Lower resolution**: Edit `config.yaml`
   ```yaml
   video:
     input_resolution:
       width: 960  # From 1280
       height: 540  # From 720
   ```
4. **Check GPU usage**: `nvidia-smi` should show python process

### Model Not Found

**Symptom**: `Failed to load X model: [Errno 2] No such file or directory`

**Solutions**:
1. Check file exists: `ls -lh models/`
2. Download missing model (see Quick Start)
3. System continues without non-critical models (plate, helmet)

### WebSocket Connection Failed

**Symptom**: Browser shows "WebSocket connection error"

**Solutions**:
1. Check server running: Look for "Application started successfully"
2. Check firewall: Allow port 8500
3. Try different port in `config.yaml`
4. Check console for errors

### OCR Not Detecting Plates

**Symptoms**: Plates detected but OCR returns no text

**Solutions**:
1. Check logs: Look for "OCR SUCCESS" or "OCR REJECT" messages
2. Lower confidence: Set `confidence_threshold: 0.0` in config
3. Check plate clarity: Plates must be readable in video
4. Verify fast-plate-ocr installed: `pip show fast-plate-ocr`

---

## Documentation

| Document | Description |
|----------|-------------|
| **README.md** | This file - Overview and quick start |
| **INSTALLATION_GUIDE.md** | Detailed installation instructions |
| **HELMET_QUICKSTART.md** | Quick start for helmet detection |
| **docs/HELMET_DETECTION.md** | Technical details of helmet detection |
| **docs/HELMET_FEATURE_SUMMARY.md** | Helmet feature implementation summary |
| **docs/TENSORRT_SETUP.md** | TensorRT optimization guide |
| **docs/PHASE_*.md** | Development phase summaries |

---

## Roadmap

### Completed
- [x] FastAPI backend with WebSocket streaming
- [x] RTSP/HLS/Video file support
- [x] YOLO11n vehicle detection
- [x] ByteTrack multi-object tracking
- [x] License plate detection
- [x] fast-plate-ocr integration (30-50x faster)
- [x] Helmet detection for motorcycles
- [x] TensorRT acceleration
- [x] Traffic counting with virtual line
- [x] Responsive web UI
- [x] Real-time statistics dashboard
- [x] License plate display panel

### 🚧 In Progress
- [ ] Helmet violation logging and alerts
- [ ] Export traffic reports (CSV/JSON)
- [ ] Multi-camera support
- [ ] Recording and playback

### 📅 Planned
- [ ] Speed estimation
- [ ] Vehicle color detection
- [ ] Violation detection (red light, wrong lane)
- [ ] Database integration (PostgreSQL)
- [ ] RESTful API for external integration
- [ ] Mobile app (iOS/Android)
- [ ] Cloud deployment support

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Ultralytics** - YOLO11 object detection
- **ByteTrack** - Multi-object tracking algorithm
- **fast-plate-ocr** - Fast license plate OCR
- **FastAPI** - Modern web framework
- **NVIDIA** - CUDA and TensorRT acceleration

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/AI_Traffic/issues)
- **Documentation**: Check `docs/` folder
- **Email**: your.email@example.com

---

## 🎉 Screenshots

### Main Interface
```
┌──────────────────────────────────────────────────────────┐
│ 🚦 AI Traffic Monitoring          [Settings] [Stats]    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│                    [Video Display]                       │
│              Real-time detection overlay                 │
│         Vehicle boxes • Plate boxes • Helmet boxes      │
│                                                          │
├──────────────────────────────────────────────────────────┤
│ FPS: 28.5 │ Vehicles: 12 │ Density: Medium │ In: 45↓ │ Out: 38↑ │
└──────────────────────────────────────────────────────────┘

Sidebar:
┌─────────────────────┐
│ LICENSE PLATES      │
├─────────────────────┤
│ Vehicle #5          │
│ [img] 30L21050      │
├─────────────────────┤
│ Vehicle #12         │
│ [img] 29H18115      │
└─────────────────────┘
```

---

**Made with ❤️ using AI and Computer Vision**

**Status**: ✅ Production Ready | **Version**: 1.0.0 | **Last Updated**: 2026-08-31

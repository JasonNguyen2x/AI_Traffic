# Performance Optimization — Traffic AI Monitoring System

> **Date:** 2026-08-28  
> **Status:** P1–P4 Applied ✅ | P5–P12 Pending

---

## Summary

Four critical performance bottlenecks were identified and fixed in the real-time video processing pipeline. These optimizations target the **per-frame hot path** — the code that runs 30 times per second on every frame.

| Fix | Description | Est. FPS Gain | Files Changed |
|-----|-------------|--------------|---------------|
| **P1** | Eliminate redundant `frame.copy()` | +3–5 FPS | `vehicle_detector.py`, `plate_detector.py` |
| **P2** | Batch GPU→CPU tensor transfers | +2–5 FPS | `vehicle_detector.py`, `plate_detector.py` |
| **P3** | Skip no-op `cv2.resize()` | +1–2 FPS | `main.py` |
| **P4** | TurboJPEG fast JPEG encoder | +2–4 FPS | `main.py`, `requirements.txt` |
| | **Total estimated gain** | **+8–16 FPS** | |

---

## P1: Eliminate Redundant `frame.copy()`

### Problem

Every draw function created a full copy of the frame before drawing bounding boxes. With 1280×720 BGR frames (~2.7 MB each), this caused:

- **2 copies per frame** in the main pipeline (`draw_detections` + `draw_plates_from_associations`)
- **~160 MB/s** of unnecessary memory allocation at 30 FPS
- Increased GC pressure and cache misses

### Before

```python
# vehicle_detector.py - draw_detections()
def draw_detections(self, frame, detections, ...):
    frame_copy = frame.copy()          # 2.7 MB allocation
    cv2.rectangle(frame_copy, ...)
    return frame_copy

# plate_detector.py - draw_plates_from_associations()
def draw_plates_from_associations(frame, ...):
    frame_copy = frame.copy()          # 2.7 MB allocation
    cv2.rectangle(frame_copy, ...)
    return frame_copy
```

### After

```python
# vehicle_detector.py - draw_detections()
def draw_detections(self, frame, detections, ...):
    # Draw directly on frame (no copy) — caller is done reading it
    cv2.rectangle(frame, ...)
    return frame

# plate_detector.py - draw_plates_from_associations()
def draw_plates_from_associations(frame, ...):
    # Draw directly on frame (no copy) — caller is done reading it
    cv2.rectangle(frame, ...)
    return frame
```

### Why This Is Safe

The frame is consumed (encoded to JPEG and sent over WebSocket) immediately after drawing. The async plate/OCR tasks that reference the same frame are safe because:

1. **YOLO inference** internally copies and preprocesses the frame (resize → normalize → tensor), so drawn boxes don't affect detection accuracy.
2. **OCR** crops a small plate region that sits inside vehicle bounding boxes — far from the drawn box edges.

### Files Changed

- `backend/ai/vehicle_detector.py` — `draw_detections()`
- `backend/ai/plate_detector.py` — `draw_plates_from_associations()`

---

## P2: Batch GPU→CPU Tensor Transfers

### Problem

Detection results were transferred from GPU to CPU **one tensor element at a time**, causing N×3 CUDA synchronization barriers per frame (where N = number of detected vehicles):

```
For 10 vehicles: 30 separate GPU→CPU transfers per frame
At 30 FPS: 900 CUDA sync points per second
```

Each `tensor[i].cpu().numpy()` call forces a CUDA stream synchronization, stalling the GPU pipeline.

### Before

```python
# vehicle_detector.py - detect()
for i in range(len(boxes)):
    box = boxes.xyxy[i].cpu().numpy()       # GPU→CPU sync #1
    conf = float(boxes.conf[i].cpu().numpy()) # GPU→CPU sync #2
    cls_id = int(boxes.cls[i].cpu().numpy())  # GPU→CPU sync #3
```

### After

```python
# vehicle_detector.py - detect()
# Batch GPU→CPU transfer (single sync instead of N×3)
all_xyxy = boxes.xyxy.cpu().numpy()   # 1 GPU→CPU transfer for ALL boxes
all_conf = boxes.conf.cpu().numpy()   # 1 GPU→CPU transfer for ALL confs
all_cls = boxes.cls.cpu().numpy().astype(int)  # 1 GPU→CPU transfer for ALL classes

for i in range(len(boxes)):
    x1, y1, x2, y2 = all_xyxy[i]     # Pure CPU numpy indexing
    conf = float(all_conf[i])
    cls_id = int(all_cls[i])
```

### Impact

| Metric | Before | After |
|--------|--------|-------|
| GPU→CPU syncs (10 vehicles) | 30/frame | 3/frame |
| GPU→CPU syncs at 30 FPS | 900/sec | 90/sec |
| Estimated time saving | — | 2–5ms/frame |

### Files Changed

- `backend/ai/vehicle_detector.py` — `detect()`
- `backend/ai/plate_detector.py` — `detect_on_frame()`

---

## P3: Skip No-Op `cv2.resize()`

### Problem

The configuration has identical input and output resolutions:

```yaml
video:
  input_resolution:
    width: 1280
    height: 720
  output_resolution:
    width: 1280
    height: 720
```

Yet `cv2.resize()` was called unconditionally on every frame — allocating a new 2.7 MB frame and running pixel interpolation for a 1280×720 → 1280×720 "resize" that changes nothing.

Additionally, `output_width` and `output_height` were re-read from the config object on every frame inside the loop.

### Before

```python
# main.py - websocket_stream() loop
output_width = config.video.output_resolution.width    # Re-read every frame
output_height = config.video.output_resolution.height  # Re-read every frame
frame_resized = cv2.resize(frame, (output_width, output_height))  # No-op!
```

### After

```python
# Cached ONCE before the loop
output_width = config.video.output_resolution.width
output_height = config.video.output_resolution.height
jpeg_quality = config.video.jpeg_quality

# Inside the loop: skip resize when dimensions already match
h, w = frame.shape[:2]
if w != output_width or h != output_height:
    frame_out = cv2.resize(frame, (output_width, output_height))
else:
    frame_out = frame  # Zero-cost reference, no allocation
```

### Files Changed

- `backend/main.py` — `websocket_stream()`

---

## P4: TurboJPEG Fast JPEG Encoder

### Problem

`cv2.imencode('.jpg', ...)` uses the standard **libjpeg** encoder, which is single-threaded and not SIMD-optimized. For 1280×720 frames at quality 75, this typically takes **3–8ms per frame**.

### After

The pipeline now tries to use **libjpeg-turbo** via the `PyTurboJPEG` wrapper, which uses SIMD (SSE2/AVX2/NEON) instructions for 2–3× faster encoding. Falls back gracefully to OpenCV if unavailable.

```python
# Initialization (once per WebSocket connection)
turbojpeg_encoder = None
try:
    from turbojpeg import TurboJPEG, TJPF_BGR
    turbojpeg_encoder = TurboJPEG()
    logger.info("Using TurboJPEG for fast JPEG encoding")
except (ImportError, OSError, RuntimeError) as e:
    logger.info(f"TurboJPEG not available ({e}), using OpenCV JPEG encoder")

# Per-frame encoding
if turbojpeg_encoder is not None:
    buffer = turbojpeg_encoder.encode(frame_out, quality=jpeg_quality)
    ret = buffer is not None
else:
    ret, buffer = cv2.imencode('.jpg', frame_out,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if ret:
        buffer = buffer.tobytes()
```

### Installation

`PyTurboJPEG` (the Python wrapper) is in `requirements.txt`. The native library must be installed separately:

```bash
# Option 1: Conda (recommended)
conda install -c conda-forge libjpeg-turbo

# Option 2: Download installer
# https://github.com/libjpeg-turbo/libjpeg-turbo/releases
```

### Expected Encoding Times

| Encoder | 1280×720 @ Q75 | 1920×1080 @ Q75 |
|---------|---------------|-----------------|
| cv2.imencode (libjpeg) | 4–8 ms | 8–15 ms |
| TurboJPEG (libjpeg-turbo) | 1.5–3 ms | 3–6 ms |

### Files Changed

- `backend/main.py` — `websocket_stream()`
- `requirements.txt` — added `PyTurboJPEG>=1.7.0`

---

## Bonus: Top-Level `import cv2`

As part of P1/P2, `import cv2` was moved from inside method bodies to the module top level in both detector files. While Python caches imports in `sys.modules`, the module lookup still has measurable overhead when called 30×/sec in a hot loop.

### Files Changed

- `backend/ai/vehicle_detector.py` — moved `import cv2` to line 5
- `backend/ai/plate_detector.py` — moved `import cv2` to line 5

---

## Remaining Optimizations (Pending)

| Priority | ID | Description | Effort |
|----------|----|-------------|--------|
| 🟡 Medium | **P5** | Throttle stats WebSocket sends (every 5 frames instead of every frame) | Trivial |
| 🟡 Medium | **P6** | Replace frame Queue with single-slot buffer (`Lock` + reference) | Low |
| 🟡 Medium | **P7** | Add ByteTrack object tracker for persistent vehicle IDs | Medium |
| 🟡 Medium | **P9** | Batch OCR processing (composite image instead of per-plate calls) | Medium |
| 🟢 Low | **P10** | Replace deprecated `asyncio.get_event_loop()` with `get_running_loop()` | Trivial |
| 🟢 Low | **P11** | Remove redundant HTTP stats polling (already pushed via WebSocket) | Trivial |
| 🟢 Low | **P12** | Stream video uploads instead of loading entire file into memory | Medium |

---

## How to Verify

After applying these changes, monitor the FPS overlay in the browser and the server logs:

```
[INFO] FPS: 32.5 | Frame: 30.8ms | Inference: 18.2ms | Draw: 1.1ms | Resize: 0.0ms | Encode: 2.3ms | Vehicles: 8
```

Key metrics to watch:
- **Resize** should show `0.0ms` when input matches output resolution
- **Encode** should drop from ~5ms to ~2ms with TurboJPEG
- **Draw** should decrease slightly without `frame.copy()` overhead
- **Overall FPS** should increase by 8–16 FPS depending on scene complexity

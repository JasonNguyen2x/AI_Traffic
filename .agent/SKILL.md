You are a Senior AI Engineer, Computer Vision Engineer, Python Backend Engineer, and Full-Stack Web Engineer.

I want you to design and implement a real-time AI web application for vehicle traffic monitoring and license plate recognition.

==================================================
1. PROJECT GOAL
==================================================

Build a web-based AI application that receives ONE network video stream and performs:

1. Vehicle detection
2. Vehicle tracking
3. Traffic counting
4. License plate detection
5. License plate OCR
6. Real-time visualization in a web browser

The application processes EXACTLY ONE stream at a time.

There is NO requirement for:
- Multiple cameras
- Multiple simultaneous streams
- Database
- User authentication
- Redis
- Kafka
- PostgreSQL
- MongoDB
- Microservices
- Kubernetes
- Docker in the first version

Keep the architecture simple and maintainable.

==================================================
2. IMPORTANT STREAM REQUIREMENT
==================================================

The input is NOT guaranteed to be RTSP.

The user may provide:

- IP / hostname
- Port
- Protocol

Possible protocols may include:

- RTSP
- HTTP/MJPEG
- TCP
- UDP
- WebSocket
- other protocols if a suitable decoder exists

IMPORTANT:

A network port alone does NOT mean that it contains a video stream.

For example:

192.168.1.100:5000

could contain:
- H264/H265 video
- JPEG frames
- custom binary data
- sensor data
- another protocol

Therefore the application must separate:

Network Transport
        ↓
Protocol
        ↓
Video Stream Format
        ↓
Video Decoder
        ↓
Video Frames
        ↓
AI Pipeline

Do NOT assume arbitrary TCP/UDP data is directly decodable as video.

If the protocol or payload format is unknown, report a clear error to the user.

==================================================
3. FINAL ARCHITECTURE
==================================================

Use this architecture:

                    WEB BROWSER
                         │
                  HTML + CSS + JS
                         │
                  HTTP + WebSocket
                         │
                         ▼
                  ┌──────────────┐
                  │   FastAPI    │
                  └──────┬───────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Stream Manager          AI Pipeline
              │                     │
              ▼                     │
       Stream Adapter               │
              │                     │
              ▼                     │
       Video Decoder               │
              │                     │
              └──────────►──────────┘
                                    │
                                    ▼
                            Vehicle Detection
                                    │
                                    ▼
                               Tracking
                                    │
                       ┌────────────┴────────────┐
                       │                         │
                       ▼                         ▼
                Traffic Counting          Vehicle Crop
                                                 │
                                                 ▼
                                        Plate Detection
                                                 │
                                                 ▼
                                            Plate Crop
                                                 │
                                                 ▼
                                              OCR
                                                 │
                                                 ▼
                                      Plate Post Processing
                                                 │
                                                 ▼
                                            WebSocket
                                                 │
                                                 ▼
                                             Browser

The browser NEVER directly connects to the camera/network port.

FastAPI/backend connects to the external network stream.

==================================================
4. TECHNOLOGY STACK
==================================================

Backend:

- Python 3.11+
- FastAPI
- Uvicorn
- OpenCV
- FFmpeg where appropriate
- PyTorch
- Ultralytics YOLO
- ByteTrack or BoT-SORT
- EasyOCR
- Pydantic
- PyYAML

Frontend:

- HTML5
- CSS3
- Vanilla JavaScript
- WebSocket API
- Fetch API

DO NOT use:

- React
- Vue
- Angular
- Next.js
- Vite
- Node.js
- npm

The frontend must be completely static HTML/CSS/JavaScript served by FastAPI.

==================================================
5. PROJECT STRUCTURE
==================================================

Use a clean structure similar to:

traffic-ai/
│
├── backend/
│   ├── main.py
│   │
│   ├── api/
│   │   ├── stream.py
│   │   ├── statistics.py
│   │   └── config.py
│   │
│   ├── stream/
│   │   ├── base.py
│   │   ├── manager.py
│   │   ├── rtsp.py
│   │   ├── http_mjpeg.py
│   │   ├── tcp.py
│   │   └── udp.py
│   │
│   ├── ai/
│   │   ├── pipeline.py
│   │   ├── vehicle_detector.py
│   │   ├── tracker.py
│   │   ├── plate_detector.py
│   │   └── ocr.py
│   │
│   ├── counting/
│   │   └── line_counter.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   └── state.py
│   │
│   └── utils/
│       ├── image.py
│       └── logging.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   └── js/
│       └── app.js
│
├── models/
│   ├── vehicle.pt
│   └── license_plate.pt
│
├── config.yaml
├── requirements.txt
├── README.md
└── .gitignore

You may modify the structure if there is a strong technical reason.

==================================================
6. CONFIG.YAML
==================================================

Use config.yaml for STATIC system configuration.

Do NOT store the runtime camera IP/port entered by the user in config.yaml.

Example:

server:
  host: 0.0.0.0
  port: 8000

ai:
  device: auto
  vehicle_confidence: 0.4
  plate_confidence: 0.4
  inference_size: 640
  plate_detection_interval: 5

models:
  vehicle: models/vehicle.pt
  plate: models/license_plate.pt

tracking:
  tracker: bytetrack.yaml

stream:
  buffer_size: 2
  reconnect_delay: 3

counting:
  line:
    x1: 100
    y1: 300
    x2: 900
    y2: 300

The user-entered runtime stream configuration should be kept in application memory only.

Example runtime data:

{
    "host": "192.168.1.100",
    "port": 5000,
    "protocol": "tcp"
}

Do not persist credentials or sensitive connection information unnecessarily.

==================================================
7. VEHICLE DETECTION
==================================================

Use Ultralytics YOLO.

Detect at least:

- car
- motorcycle
- bus
- truck

Each detection must contain:

- bounding box
- confidence
- class
- tracking ID after tracking

Example:

{
    "track_id": 27,
    "class": "car",
    "confidence": 0.94,
    "bbox": [x1, y1, x2, y2]
}

The vehicle model path must come from config.yaml.

Do not hardcode a model filename in Python.

==================================================
8. OBJECT TRACKING
==================================================

Use ByteTrack or BoT-SORT.

Tracking must be separated from detection.

The same vehicle should maintain a stable ID across frames whenever possible.

Example:

Frame 100:
Car ID 15

Frame 101:
Car ID 15

Frame 102:
Car ID 15

Do not count every detection as a new vehicle.

==================================================
9. TRAFFIC COUNTING
==================================================

Implement virtual line crossing.

Example:

(x1, y1) ---------------- (x2, y2)

When a tracked vehicle crosses the line:

count += 1

Support:

- IN
- OUT

Statistics:

- Total vehicles
- Cars
- Motorcycles
- Trucks
- Buses
- IN
- OUT

The system must prevent double counting using tracking IDs.

The counting line must be configurable.

For the first version, it can be configured in config.yaml.

Later it may be moved to the UI.

==================================================
10. LICENSE PLATE DETECTION
==================================================

Use a separate YOLO model for license plate detection.

Pipeline:

Vehicle Detection
        ↓
Tracking
        ↓
Vehicle Crop
        ↓
License Plate Detection
        ↓
Plate Crop
        ↓
Preprocessing
        ↓
OCR
        ↓
Post Processing
        ↓
License Plate Result

Do not run plate detection/OCR unnecessarily on every frame.

Use a configurable interval.

Example:

plate_detection_interval: 5

==================================================
11. OCR
==================================================

Use EasyOCR.

The target use case is Vietnamese license plates.

Keep:

raw OCR result

separate from:

normalized license plate result

Example:

raw:
30A12345

normalized:
30A-123.45

Implement a post-processing layer.

Do not blindly replace characters such as:

0/O
1/I
5/S
8/B

Use confidence and plate-format validation where possible.

==================================================
12. TEMPORAL OCR
==================================================

Do not trust a single OCR frame.

For a tracked vehicle:

Vehicle ID 27

Frame 1:
30A12345

Frame 5:
30A12345

Frame 10:
30A1234S

Frame 15:
30A12345

Use temporal voting/confidence aggregation.

Return the most reliable result.

Each tracked vehicle should maintain temporary OCR state in memory.

No database is needed.

==================================================
13. REAL-TIME PROCESSING
==================================================

This is a real-time application.

Prioritize LOW LATENCY.

Do not allow an unlimited frame queue.

Use a bounded buffer.

Example:

buffer_size = 2

If AI processing is slower than incoming frames:

- discard old frames
- process the newest available frame
- avoid latency continuously increasing

Architecture:

Stream Receiver
      ↓
Latest Frame Buffer
      ↓
AI Worker

Do not block stream reception while AI inference is running.

==================================================
14. WEB VIDEO DISPLAY
==================================================

The browser must display the AI-processed frames.

For the first version:

Backend:

OpenCV frame
    ↓
JPEG encoding
    ↓
WebSocket binary message

Frontend:

WebSocket
    ↓
Blob
    ↓
Image element / Canvas

The WebSocket may send two types of information:

1. Processed JPEG frame
2. JSON metadata/statistics

Example metadata:

{
    "type": "stats",
    "total": 128,
    "cars": 82,
    "motorcycles": 39,
    "trucks": 7,
    "buses": 0,
    "in": 72,
    "out": 56,
    "latest_plate": "30A-123.45"
}

Design the WebSocket layer so that WebRTC could be introduced later if necessary.

Do NOT implement WebRTC in version 1.

==================================================
15. FRONTEND
==================================================

Use ONLY:

- index.html
- style.css
- app.js

No framework.

The interface should be simple but professional.

Main UI:

----------------------------------------------
TRAFFIC AI
----------------------------------------------

STREAM CONNECTION

Protocol:
[ TCP ▼ ]

Host / IP:
[ 192.168.1.100 ]

Port:
[ 5000 ]

[ CONNECT ] [ DISCONNECT ]

Status:
● DISCONNECTED

----------------------------------------------

LIVE VIDEO

+------------------------------------------+
|                                          |
|          AI PROCESSED VIDEO              |
|                                          |
|   Vehicle bounding boxes                 |
|   Tracking IDs                           |
|   License plates                         |
|   Counting line                          |
|                                          |
+------------------------------------------+

----------------------------------------------

TRAFFIC STATISTICS

Total Vehicles: 128

Cars:          82
Motorcycles:   39
Trucks:         7
Buses:          0

IN:            72
OUT:           56

----------------------------------------------

LATEST LICENSE PLATES

30A-123.45
29B-456.78
18C-123.45

----------------------------------------------

The UI should update without page reload.

==================================================
16. FRONTEND DESIGN
==================================================

Use clean modern CSS.

Requirements:

- Responsive layout
- Dark dashboard style is acceptable
- Clear connection status
- Clear error messages
- Large video area
- Statistics cards
- Latest license plate section
- Connect/disconnect buttons
- Disable Connect while connecting
- Disable Disconnect when disconnected

Do not make the UI overly complicated.

No frontend build process.

Opening:

http://localhost:8000

must load the application.

==================================================
17. FASTAPI ROUTES
==================================================

Implement:

GET /

POST /api/stream/connect

POST /api/stream/disconnect

GET /api/stream/status

GET /api/statistics

GET /api/config

WebSocket:

/ws/stream

Connection request:

POST /api/stream/connect

Example JSON:

{
    "protocol": "tcp",
    "host": "192.168.1.100",
    "port": 5000
}

The backend must validate the input.

==================================================
18. STREAM MANAGER
==================================================

Create a StreamManager responsible for:

- starting the stream
- stopping the stream
- reconnecting
- exposing the latest frame
- tracking connection state
- preventing duplicate stream workers
- graceful shutdown

Only ONE stream can be active.

If the user connects while another stream is active:

Either:
- reject the request with a clear error

OR:
- safely disconnect the previous stream first

Choose the cleaner approach and explain it.

==================================================
19. STREAM ADAPTER DESIGN
==================================================

Create an interface such as:

BaseStreamAdapter

with methods conceptually similar to:

connect()
read_frame()
is_connected()
disconnect()

Implement adapters where technically possible:

RTSPAdapter
HTTPMJPEGAdapter
TCPAdapter
UDPAdapter

IMPORTANT:

Do NOT pretend that raw TCP or UDP automatically contains video.

For TCP/UDP:

Transport layer:
    TCP/UDP socket

must be separated from:

Payload:
    H264/H265/JPEG/custom protocol/etc.

If the payload format is unknown, the adapter must report:

"Connected to TCP port, but video payload format is unknown."

The system should be designed so a custom decoder can later be added.

==================================================
20. ERROR HANDLING
==================================================

The backend must handle:

- invalid IP
- invalid port
- connection timeout
- connection refused
- authentication failure
- unsupported protocol
- invalid video stream
- decoder error
- stream disconnect
- AI model loading failure
- CUDA unavailable
- OCR failure

Frontend status examples:

DISCONNECTED
CONNECTING
CONNECTED
RECONNECTING
STREAM ERROR
DECODER ERROR
AI ERROR

Do not expose sensitive credentials in error messages or logs.

==================================================
21. GPU SUPPORT
==================================================

Use NVIDIA GPU when available.

Detect CUDA automatically.

Configuration:

device: auto

Possible behavior:

auto:
    CUDA available → cuda
    CUDA unavailable → cpu

Allow explicit:

device: cuda

or:

device: cpu

Use FP16 when supported.

Do not crash if CUDA is unavailable.

==================================================
22. PERFORMANCE
==================================================

Optimize for a single real-time stream.

Implement:

- frame skipping
- latest-frame strategy
- bounded buffer
- GPU inference
- FP16 when possible
- configurable resolution
- separate receiving and AI processing
- temporal OCR
- configurable plate detection interval

The system should favor low latency over processing every frame.

==================================================
23. MEMORY MANAGEMENT
==================================================

Because there is no database, all runtime state is in memory.

Prevent:

- unlimited OCR history
- unlimited tracked vehicle history
- unlimited frame queues
- memory leaks

When a vehicle disappears for a configurable amount of time, remove its temporary state.

Keep only the necessary recent statistics.

==================================================
24. LOGGING
==================================================

Use Python logging.

Example:

[INFO] Starting application
[INFO] Loading vehicle model
[INFO] Loading plate model
[INFO] Initializing OCR
[INFO] CUDA available
[INFO] Connecting to stream
[INFO] Stream connected
[INFO] Video resolution: 1920x1080
[INFO] FPS: 25
[INFO] Vehicle tracking started
[WARNING] Stream disconnected
[INFO] Reconnecting

Do not log passwords.

==================================================
25. NO DATABASE
==================================================

Do NOT create any database layer.

Do NOT create:

- SQLAlchemy
- PostgreSQL
- MongoDB
- database migrations

All information is runtime-only.

If the application restarts:

- counters reset
- tracked vehicles reset
- OCR state resets

This is acceptable.

==================================================
26. SECURITY
==================================================

The browser should send connection information to FastAPI.

The browser should NOT directly access the camera.

The backend connects to the camera/network service.

Do not expose credentials unnecessarily.

Do not log passwords.

Validate host and port input.

==================================================
27. TESTABILITY
==================================================

Create components that can be tested independently.

For example:

StreamAdapter
VehicleDetector
Tracker
LineCounter
PlateDetector
OCRProcessor
AIPipeline

The line counter should be testable without a real camera.

The OCR post-processing should be testable with sample strings.

==================================================
28. DEVELOPMENT PHASES
==================================================

Do NOT generate the entire project in one enormous response.

Implement the project incrementally.

PHASE 1:
Architecture and technical analysis.

PHASE 2:
FastAPI + HTML/CSS/JS basic web interface.

PHASE 3:
Stream abstraction and stream receiver.

PHASE 4:
YOLO vehicle detection.

PHASE 5:
ByteTrack/BoT-SORT tracking.

PHASE 6:
Traffic line counting.

PHASE 7:
License plate detection.

PHASE 8:
EasyOCR.

PHASE 9:
Complete AI pipeline.

PHASE 10:
WebSocket real-time video and statistics.

PHASE 11:
Performance optimization.

PHASE 12:
Testing and debugging.

At each phase:

1. Explain the goal.
2. Show the files being created or modified.
3. Provide COMPLETE code for those files.
4. Explain dependencies.
5. Explain how to run.
6. Explain how to test.
7. Do not modify unrelated files.
8. Wait for confirmation before moving to the next major phase.

==================================================
29. DEVELOPMENT RULE
==================================================

Do not use placeholder pseudocode when implementation is requested.

When writing code:

- provide complete runnable files
- include imports
- include error handling
- include type hints
- use clear naming
- keep functions reasonably small
- avoid unnecessary abstraction
- avoid overengineering

If a library API has changed, use the currently correct API.

==================================================
30. FIRST TASK
==================================================

START WITH PHASE 1 ONLY.

Do NOT write the implementation yet.

Analyze:

1. Complete system architecture.
2. Data flow.
3. Stream abstraction.
4. How TCP/UDP/HTTP/RTSP should be handled.
5. How arbitrary IP + port should be validated.
6. How to determine whether a port actually carries video.
7. How video decoding should be separated from network transport.
8. AI pipeline architecture.
9. Vehicle tracking architecture.
10. Traffic counting architecture.
11. License plate detection architecture.
12. OCR architecture.
13. WebSocket architecture.
14. HTML/CSS/Vanilla JS frontend architecture.
15. Runtime memory management without a database.
16. GPU inference strategy.
17. Performance bottlenecks.
18. Potential failure cases.
19. Recommended project structure.
20. Exact information you need from me about the network stream before implementing the StreamAdapter.

Do not write code in Phase 1.

Wait for my confirmation before starting Phase 2.
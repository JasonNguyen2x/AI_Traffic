# TensorRT Setup Guide

## 📦 Prerequisites

1. **NVIDIA GPU** with Compute Capability 7.0+ (RTX 5050 ✅)
2. **CUDA 12.x** installed
3. **cuDNN 8.x** installed
4. **Python 3.11+**

## 🚀 Installation Steps

### Step 1: Install TensorRT (Option A - Recommended)

Via pip (easiest):
```bash
pip install tensorrt
```

### Step 1: Install TensorRT (Option B - Manual)

Download from NVIDIA:
1. Go to https://developer.nvidia.com/tensorrt
2. Download TensorRT for Windows
3. Extract and add to PATH

### Step 2: Verify Installation

```bash
python -c "import tensorrt; print(tensorrt.__version__)"
```

Should output: `10.0.0` or higher

### Step 3: Export Models to TensorRT

```bash
cd AI_Traffic
python setup_tensorrt.py
```

This will:
- ✅ Export `yolo11n.pt` → `yolo11n.engine` (1280×1280)
- ✅ Export `license_plate_detector.pt` → `license_plate_detector.engine` (640×640)
- ✅ Test inference on both models

**Expected output:**
```
============================================================
EXPORTING VEHICLE DETECTION MODEL
============================================================
INFO: Loading model: models/yolo11n.pt
INFO: Exporting to TensorRT...
  - Input size: 1280
  - Half precision: True
  - Workspace: 4GB
...
✅ Export complete: models/yolo11n.engine
✅ Inference test passed!

============================================================
EXPORTING PLATE DETECTION MODEL
============================================================
...
✅ ALL MODELS EXPORTED SUCCESSFULLY
```

**Time:** ~5-10 minutes per model

### Step 4: Update Config

Config already updated! Check `config.yaml`:
```yaml
models:
  vehicle: models/yolo11n.engine  # TensorRT
  vehicle_fallback: models/yolo11n.pt  # Fallback
  plate: license_plate_detector.engine
  plate_fallback: license_plate_detector.pt
```

### Step 5: Run Application

```bash
python -m backend.main
```

Look for these logs:
```
INFO: ✅ Loaded TensorRT engine: models/yolo11n.engine
INFO: ✅ Loaded TensorRT engine: models/license_plate_detector.engine
```

## 🎯 Expected Performance

### Before TensorRT (.pt models):
- **Resolution:** 640×480 → YOLO → upscale to 1280×720
- **FPS:** ~15-20
- **Inference:** ~30-40ms

### After TensorRT (.engine models):
- **Resolution:** 1280×720 → YOLO (native!) → no upscale
- **FPS:** ~25-30
- **Inference:** ~15-20ms

**Improvements:**
- ✅ **2x faster** inference
- ✅ **Native resolution** (no upscaling)
- ✅ **Better quality** (higher resolution input)
- ✅ **Same GPU memory**

## 🔧 Troubleshooting

### Error: "No module named 'tensorrt'"

**Solution:**
```bash
pip install --upgrade tensorrt
```

### Error: "CUDA not available"

**Solution:**
1. Check CUDA installation: `nvcc --version`
2. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reinstall PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### Error: "Export failed"

**Solution:**
1. Reduce workspace size: `workspace=2` in setup_tensorrt.py
2. Use smaller imgsz: `imgsz=640` instead of 1280
3. Disable FP16: `half=False`

### Fallback to .pt models

If TensorRT export fails, the system will automatically use .pt models. Check logs:
```
WARNING: TensorRT engine not found: models/yolo11n.engine
INFO: Using fallback model: models/yolo11n.pt
```

## 📊 Benchmark

Run benchmark:
```bash
python performance_test.py
```

Compare:
- **.pt model** at 640×480
- **.engine model** at 1280×720

## 🎬 Summary

1. ✅ Install TensorRT: `pip install tensorrt`
2. ✅ Export models: `python setup_tensorrt.py`
3. ✅ Run application: `python -m backend.main`
4. ✅ Enjoy 2x faster + native resolution! 🚀

## 💡 Tips

- **First run:** Export takes 5-10 min (one-time)
- **Subsequent runs:** Instant load
- **.engine files:** Tied to GPU, don't share between machines
- **Model updates:** Re-export after changing .pt models

---

**Need help?** Check logs or run with `--log-level DEBUG`

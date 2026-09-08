"""Script to export YOLO models to TensorRT format."""
import logging
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_gpu_compatibility():
    """Check GPU compatibility and print device info."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available!")
        logger.error("TensorRT requires NVIDIA GPU with CUDA support")
        return False
    
    device_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    cuda_version = torch.version.cuda
    
    logger.info("=" * 60)
    logger.info("GPU INFORMATION")
    logger.info("=" * 60)
    logger.info(f"Device: {device_name}")
    logger.info(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
    logger.info(f"CUDA Version: {cuda_version}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info("=" * 60)
    
    return True


def export_to_tensorrt(
    model_path: str,
    export_path: str = None,
    imgsz: int = 1280,
    half: bool = True,
    workspace: int = 4,  # GB
    device: int = 0,
):
    """
    Export YOLO model to TensorRT engine.
    
    Args:
        model_path: Path to .pt model
        export_path: Output path (optional, auto-generated if None)
        imgsz: Input size for inference
        half: Use FP16 precision (faster)
        workspace: Max workspace size in GB
        device: GPU device ID
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    logger.info(f"Exporting to TensorRT...")
    logger.info(f"  - Input size: {imgsz}×{imgsz}")
    logger.info(f"  - Half precision (FP16): {half}")
    logger.info(f"  - Workspace: {workspace}GB")
    logger.info(f"  - Device: cuda:{device}")
    
    # Export
    try:
        engine_path = model.export(
            format='engine',
            imgsz=imgsz,
            half=half,
            workspace=workspace,
            device=device,
            verbose=True,
            simplify=True,  # ONNX graph optimization
        )
        
        logger.info(f"Export complete: {engine_path}")
        
        # Test inference
        logger.info("Testing inference on GPU...")
        import numpy as np
        
        # Create test frame
        if imgsz == 1280:
            test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        else:
            test_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        
        # Load engine and test
        model_trt = YOLO(engine_path)
        results = model_trt.predict(test_frame, verbose=False, device=device)
        
        logger.info(f"Inference test passed!")
        logger.info(f"Detections: {len(results[0].boxes) if results[0].boxes else 0}")
        logger.info(f"Engine ready: {engine_path}")
        
        return engine_path
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise


def delete_old_engines():
    """Delete old TensorRT engines before rebuild."""
    models_dir = Path("models")
    engine_files = list(models_dir.glob("*.engine"))
    
    if not engine_files:
        logger.info("No old engines found")
        return
    
    logger.info(f"Found {len(engine_files)} old engine(s):")
    for engine in engine_files:
        logger.info(f"  - {engine.name}")
    
    response = input("\nDelete old engines? (y/n): ").lower().strip()
    
    if response == 'y':
        for engine in engine_files:
            try:
                engine.unlink()
                logger.info(f"Deleted: {engine.name}")
            except Exception as e:
                logger.error(f"Failed to delete {engine.name}: {e}")
    else:
        logger.info("Keeping old engines (will be overwritten)")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TENSORRT ENGINE BUILDER FOR RTX 3080 Ti")
    logger.info("=" * 60)
    
    # Check GPU
    if not check_gpu_compatibility():
        logger.error("GPU check failed. Exiting.")
        sys.exit(1)
    
    # Delete old engines
    print()
    delete_old_engines()
    print()
    
    # Export vehicle detection model
    logger.info("=" * 60)
    logger.info("STEP 1: EXPORTING VEHICLE DETECTION MODEL (YOLO11n)")
    logger.info("=" * 60)
    
    vehicle_model = "models/yolo11n.pt"
    vehicle_engine = None
    
    if Path(vehicle_model).exists():
        try:
            vehicle_engine = export_to_tensorrt(
                vehicle_model,
                imgsz=1280,
                half=True,
                workspace=4,
                device=0
            )
        except Exception as e:
            logger.error(f"Failed to export vehicle model: {e}")
            logger.info("\nTROUBLESHOOTING:")
            logger.info("1. Make sure CUDA toolkit is installed")
            logger.info("2. Check NVIDIA driver is up to date")
            logger.info("3. Verify TensorRT is installed: pip list | findstr tensorrt")
            sys.exit(1)
    else:
        logger.error(f"Model not found: {vehicle_model}")
        logger.info("Download model first:")
        logger.info("  from ultralytics import YOLO")
        logger.info("  model = YOLO('yolo11n.pt')")
        sys.exit(1)
    
    print()
    
    # Export plate detection model
    logger.info("=" * 60)
    logger.info("STEP 2: EXPORTING PLATE DETECTION MODEL")
    logger.info("=" * 60)
    
    plate_model = "models/license_plate_detector.pt"
    plate_engine = None
    
    if Path(plate_model).exists():
        try:
            plate_engine = export_to_tensorrt(
                plate_model,
                imgsz=640,  # Plates use smaller size
                half=True,
                workspace=4,
                device=0
            )
        except Exception as e:
            logger.error(f"Failed to export plate model: {e}")
            sys.exit(1)
    else:
        logger.warning(f"Plate model not found: {plate_model}")
        logger.warning("Skipping plate detector export")
    
    print()
    
    # Export helmet detection model
    logger.info("=" * 60)
    logger.info("STEP 3: EXPORTING HELMET DETECTION MODEL")
    logger.info("=" * 60)
    
    helmet_model = "models/helmet_detector.pt"
    helmet_engine = None
    
    if Path(helmet_model).exists():
        try:
            helmet_engine = export_to_tensorrt(
                helmet_model,
                imgsz=640,  # Helmet detection uses 640
                half=True,
                workspace=4,
                device=0
            )
        except Exception as e:
            logger.error(f"Failed to export helmet model: {e}")
            logger.warning("Continuing without helmet engine (will use .pt fallback)")
    else:
        logger.warning(f"Helmet model not found: {helmet_model}")
        logger.warning("Skipping helmet detector export")
    
    print()
    
    # Summary
    logger.info("=" * 60)
    logger.info("TENSORRT ENGINE BUILD COMPLETE")
    logger.info("=" * 60)
    
    if vehicle_engine:
        logger.info(f"Vehicle engine: {vehicle_engine}")
        logger.info(f"  Size: {Path(vehicle_engine).stat().st_size / 1024 / 1024:.1f} MB")
    
    if plate_engine:
        logger.info(f"Plate engine:   {plate_engine}")
        logger.info(f"  Size: {Path(plate_engine).stat().st_size / 1024 / 1024:.1f} MB")
    
    if helmet_engine:
        logger.info(f"Helmet engine:  {helmet_engine}")
        logger.info(f"  Size: {Path(helmet_engine).stat().st_size / 1024 / 1024:.1f} MB")
    
    logger.info("\nEngines are built for:")
    logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"  Compute: {torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}")
    
    logger.info("\n IMPORTANT:")
    logger.info("These engines ONLY work on this GPU architecture!")
    logger.info("If you switch GPU, you must rebuild engines.")
    
    logger.info("\nYou can now start the server:")
    logger.info("  python -m backend.main")


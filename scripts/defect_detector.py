# scripts to run

# Opening Environment   - .\defect_env_gpu311\Scripts\activate

# Directing to scripts - cd scripts

# Detecting defects - python defect_detector.py --mode live --model runs/detect/my_first_wandb_run/weights/best.pt --config config/system_config.json --conf 0.03

# Detecting Defect with greyscale - python defect_detector_overlay_greyscale_switch.py --mode live --model runs/detect/models/my_first_wandb_run6/weights/best.pt --display grey


# Detecting live - python defect_detector.py --mode live --model runs/detect/models/my_first_wandb_run6/weights/best.pt 

# Detecting Image - python defect_detector.py --mode image --model runs/detect/models/my_first_wandb_run6/weights/best.pt --source C:\Users\RohithSuryaCKM\Downloads\Projects\Image_detection\scripts\test_images\test_image_1.jpg

# detecting video - python defect_detector.py --mode video --model runs/detect/models/my_first_wandb_run6/weights/best.pt --source C:\Users\RohithSuryaCKM\Downloads\Projects\Image_detection\scripts\test_images\test_video_1.mp4
"""
Defect Detection - Production Inference Pipeline
Real-time defect detection using trained YOLOv11m model with Basler camera
"""

import os
import json
import time
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# Import camera module
from camera_capture import get_camera, BaslerCamera, MockCamera, convert_to_greyscale


class DefectDetector:
    """Production defect detection system"""
    
    def __init__(
        self,
        model_path: str,
        config_path: str = None,
        conf_threshold: float = 0.03,
        iou_threshold: float = 0.2,
        device: str = '0'
    ):
        """
        Initialize defect detector
        
        Args:
            model_path: Path to trained YOLOv11 model weights
            config_path: Path to system_config.json
            conf_threshold: Minimum confidence for detection
            iou_threshold: IoU threshold for NMS
            device: GPU device ('0', '1', 'cpu')
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics not installed")
        
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Load model
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names

        # Auto-detect whether this is a detection or segmentation model
        self.is_seg = getattr(self.model, 'task', 'detect') == 'segment'
        print(f"Model type: {'SEGMENTATION (mask)' if self.is_seg else 'DETECTION (bbox)'}")

        # Warmup model
        self._warmup()

        print(f"Detector initialized. Classes: {list(self.class_names.values())}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        default_config = {
            "model": {
                "confidence_threshold": 0.05,
                "iou_threshold": 0.20,
                "image_size": 640
            },
            "inspection": {
                "save_images": True,
                "save_path": "logs/inspections",
                "log_to_database": True,
                "database_path": "logs/inspections.db"
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        return default_config
    
    def _warmup(self, iterations: int = 3):
        """Warmup model with dummy inference"""
        print("Warming up model...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.model(dummy, verbose=False)
        print("Warmup complete")
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Run defect detection on image.
        The frame is first converted to rust-aware greyscale (CLAHE + rust
        darkening) — matching the preprocessing used during training —
        before being passed to the YOLO model.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            dict with detection results
        """
        start_time = time.time()
        
        # ── Greyscale pre-processing (same algorithm as camera_capture) ──
        grey_image = convert_to_greyscale(image)
        
        # Run inference on greyscale frame
        results = self.model(
            grey_image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Parse results
        defects = []
        for result in results:
            masks = result.masks  # None for detection models
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls)
                defect = {
                    'class':          self.class_names[class_id],
                    'class_id':       class_id,
                    'confidence':     round(float(box.conf), 4),
                    'bbox':           [round(x, 2) for x in box.xyxy.tolist()[0]],
                    'bbox_normalized':[round(x, 4) for x in box.xywhn.tolist()[0]],
                    'mask_polygon':   None,
                }
                # Extract mask polygon for segmentation models
                if masks is not None and i < len(masks):
                    try:
                        xy = masks.xy[i]          # (N, 2) array of pixel coords
                        defect['mask_polygon'] = xy.tolist()
                    except Exception:
                        pass
                defects.append(defect)
        
        # Sort by confidence (highest first)
        defects.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'is_defective': len(defects) > 0,
            'defect_count': len(defects),
            'defects': defects,
            'inference_time_ms': round(inference_time_ms, 2),
            'timestamp': datetime.now().isoformat()
        }
    
    def detect_from_file(self, image_path: str) -> Dict:
        """Load image from file and run detection (greyscale pre-processing applied automatically)."""
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f'Failed to load image: {image_path}'}
        
        result = self.detect(image)   # greyscale conversion happens inside detect()
        result['image_path'] = image_path
        return result
    
    def detect_from_video(self, video_path: str, output_path: Optional[str] = None):
        """Run detection on a video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        out = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        print(f"Processing video: {video_path}")
        print("Press 'q' to quit early.")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to greyscale before inference; draw results on original frame
                result = self.detect(frame)          # greyscale applied inside detect()
                annotated = self.draw_results(frame, result)
                
                if out is not None:
                    out.write(annotated)
                
                cv2.imshow("Defect Detection - Video", cv2.resize(annotated, (800, 600)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
    
    def draw_results(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """
        Draw detection results on image
        
        Args:
            image: Original image
            result: Detection result dict
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]
        
        # Scale text/thickness based on image size (reference: 640px)
        scale = max(w, h) / 640.0
        font_scale = 0.3 * scale
        thickness = max(1, int(2 * scale))
        box_thickness = max(2, int(3 * scale))
        label_pad = int(10 * scale)
        
        colors = {
            's': (255, 255, 255),   # Green
        }
        default_color = (0, 255, 255)
        
        for defect in result.get('defects', []):
            bbox = defect['bbox']
            x1, y1, x2, y2 = [int(x) for x in bbox]
            color = colors.get(defect['class'].lower(), default_color)

            mask_poly = defect.get('mask_polygon')
            if mask_poly and len(mask_poly) >= 3:
                # ── Segmentation model: draw filled semi-transparent mask ──
                pts = np.array(mask_poly, dtype=np.int32)
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)
                cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=box_thickness)
            else:
                # ── Detection model: draw plain bounding box ──
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

            # Draw label (same for both modes)
            label = f"{defect['class']}: {defect['confidence']:.1%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - label_pad), (x1 + label_size[0] + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - int(label_pad / 2)),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        # Draw overall status
        status = " " if result['is_defective'] else " "
        status_color = (0, 0, 255) if result['is_defective'] else (0, 255, 0)
        status_scale = 1.5 * scale
        status_thick = max(2, int(3 * scale))
        cv2.putText(annotated, status, (10, int(50 * scale)),
                   cv2.FONT_HERSHEY_SIMPLEX, status_scale, status_color, status_thick)
        
        # Draw inference time
        cv2.putText(annotated, f"{result['inference_time_ms']:.1f}ms", (10, int(90 * scale)),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return annotated


class InspectionLogger:
    """Log inspection results to SQLite database"""
    
    def __init__(self, db_path: str = "logs/inspections.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Create database schema"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_path TEXT,
                result TEXT,
                defect_count INTEGER,
                defects TEXT,
                max_confidence REAL,
                inference_time_ms REAL,
                shift TEXT,
                batch TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def log(self, result: Dict, image_path: str = None, 
            shift: str = None, batch: str = None):
        """Log inspection result to database"""
        conn = sqlite3.connect(self.db_path)
        
        max_conf = max([d['confidence'] for d in result['defects']]) if result['defects'] else 0
        
        conn.execute('''
            INSERT INTO inspections 
            (timestamp, image_path, result, defect_count, defects, 
             max_confidence, inference_time_ms, shift, batch)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['timestamp'],
            image_path,
            'REJECT' if result['is_defective'] else 'PASS',
            result['defect_count'],
            json.dumps(result['defects']),
            max_conf,
            result['inference_time_ms'],
            shift,
            batch
        ))
        
        conn.commit()
        conn.close()
    
    def get_stats(self, hours: int = 24) -> Dict:
        """Get inspection statistics for the last N hours"""
        conn = sqlite3.connect(self.db_path)
        
        # Total counts
        results = conn.execute('''
            SELECT result, COUNT(*) FROM inspections 
            WHERE datetime(timestamp) > datetime('now', ?)
            GROUP BY result
        ''', (f'-{hours} hours',)).fetchall()
        
        stats = {result: count for result, count in results}
        total = sum(stats.values())
        
        conn.close()
        
        return {
            'period_hours': hours,
            'total_inspections': total,
            'passed': stats.get('PASS', 0),
            'rejected': stats.get('REJECT', 0),
            'defect_rate': stats.get('REJECT', 0) / max(1, total)
        }


class ProductionInspectionSystem:
    """Complete production inspection system with camera integration"""
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "config/system_config.json",
        conf_threshold: float = 0.01,
        use_mock_camera: bool = False
    ):
        """
        Initialize production inspection system
        
        Args:
            model_path: Path to trained model
            config_path: Path to system configuration
            conf_threshold: Confidence threshold for detection
            use_mock_camera: Use mock camera for testing
        """
        self.detector = DefectDetector(model_path, config_path, conf_threshold=conf_threshold)
        self.camera = get_camera(config_path, use_mock=use_mock_camera)
        self.logger = InspectionLogger()
        
        self._running = False
    
    def inspect_once(self, save_image: bool = True) -> Dict:
        """
        Run single inspection cycle
        
        Returns:
            Inspection result
        """
        # Capture image
        image = self.camera.capture()
        if image is None:
            return {'error': 'Camera capture failed'}
        
        # Run detection
        result = self.detector.detect(image)
        
        # Save image if requested
        if save_image:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            status = 'reject' if result['is_defective'] else 'pass'
            
            save_dir = Path("logs/inspections") / status
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save annotated image
            annotated = self.detector.draw_results(image, result)
            image_path = save_dir / f"{status}_{timestamp}.png"
            cv2.imwrite(str(image_path), annotated)
            
            result['saved_image'] = str(image_path)
        
        # Log to database
        self.logger.log(result, result.get('saved_image'))
        
        return result
    
    def print_result(self, result: Dict):
        """Print formatted inspection result"""
        if result.get('error'):
            print(f"❌ Error: {result['error']}")
            return
        
        if result['is_defective']:
            print(f"⛔ REJECT - {result['defect_count']} defect(s) detected")
            for d in result['defects']:
                print(f"   └─ {d['class']}: {d['confidence']:.1%}")
        else:
            print(f"✅ PASS - No defects detected")
        
        print(f"   Inference: {result['inference_time_ms']:.1f}ms")
    
    def start(self):
        """Start continuous inspection mode with live video feed"""
        if not self.camera.connect():
            print("Failed to connect to camera")
            return

        # Start continuous grabbing for high-FPS inference.
        # capture() will now just retrieve the already-buffered latest frame
        # instead of restarting the hardware stream on every single frame.
        self.camera.start_streaming()
        
        self._running = True
        print("\n" + "="*50)
        print("LIVE INSPECTION MODE")
        print("Press 'q' to quit | 's' to save snapshot | 'v' to toggle greyscale view")
        print("="*50 + "\n")
        
        frame_count = 0
        fps_start = time.time()
        display_fps = 0.0
        show_grey = False          # V-key toggle: False = colour, True = greyscale
        
        try:
            while self._running:
                # Capture image
                image = self.camera.capture()
                if image is None:
                    print("Camera capture failed, retrying...")
                    time.sleep(0.5)
                    continue
                 # Resize image first so the CPU doesn't choke on the greyscale filter
                image = cv2.resize(image, (1280, 1280))
                # Run detection (always on greyscale internally)
                result = self.detector.detect(image)

                # Decide which frame to annotate for display
                if show_grey:
                    # Convert to greyscale and back to BGR so coloured overlays still work
                    from camera_capture import convert_to_greyscale
                    grey_bgr = convert_to_greyscale(image)   # returns 3-ch BGR-grey
                    display_frame = grey_bgr
                else:
                    display_frame = image

                # Draw bounding boxes on the chosen display frame
                annotated = self.detector.draw_results(display_frame, result)
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    display_fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.time()
                
                # Draw FPS and stats on frame
                cv2.putText(annotated, f"FPS: {display_fps:.1f}", (10, annotated.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated, f"Defects: {result['defect_count']}", (10, annotated.shape[0] - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # HUD: display mode indicator (top-right corner)
                mode_label = "[V] GREYSCALE" if show_grey else "[V] COLOUR"
                mode_color = (200, 200, 200) if show_grey else (0, 200, 255)
                (lw, lh), _ = cv2.getTextSize(mode_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.putText(annotated, mode_label,
                            (annotated.shape[1] - lw - 10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
                
                # Show live feed
                cv2.namedWindow("Defect Detection - Live Feed", cv2.WINDOW_NORMAL)
                cv2.imshow("Defect Detection - Live Feed", annotated)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting live feed...")
                    break
                elif key == ord('v'):
                    show_grey = not show_grey
                    label = "GREYSCALE" if show_grey else "COLOUR"
                    print(f"  [V] Display mode switched to: {label}")
                elif key == ord('s'):
                    # Save snapshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    status = 'reject' if result['is_defective'] else 'pass'
                    save_dir = Path("logs/inspections") / status
                    save_dir.mkdir(parents=True, exist_ok=True)
                    snap_path = save_dir / f"snapshot_{status}_{timestamp}.png"
                    cv2.imwrite(str(snap_path), annotated)
                    print(f"  Snapshot saved: {snap_path}")
                
                # Log to database
                # if result['is_defective']:
                #     self.logger.log(result)
                
        except KeyboardInterrupt:
            print("\nStopping inspection...")
        finally:
            self.camera.stop_streaming()   # Stop continuous grab before cleanup
            cv2.destroyAllWindows()
            self.stop()
    
    def stop(self):
        """Stop inspection and cleanup"""
        self._running = False
        self.camera.disconnect()
        
        # Print final statistics
        stats = self.logger.get_stats(hours=24)
        print("\n" + "="*50)
        print("SESSION STATISTICS (Last 24 hours)")
        print("="*50)
        print(f"Total inspections: {stats['total_inspections']}")
        print(f"Passed: {stats['passed']}")
        print(f"Rejected: {stats['rejected']}")
        print(f"Defect rate: {stats['defect_rate']:.2%}")
        print("="*50)


def main():
    """Main entry point for production inference"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Defect Detection Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  live    - Live camera feed with bounding boxes (press 'q' to quit, 's' to snapshot)
  image   - Capture single image from camera, show detection result
  video   - Process a video file (.mp4) with detection

Examples:
  python defect_detector.py --mode live --model path/to/best.pt
  python defect_detector.py --mode image --model path/to/best.pt
  python defect_detector.py --mode video --model path/to/best.pt --source video.mp4
  python defect_detector.py --mode image --model path/to/best.pt --source photo.jpg
        """
    )
    parser.add_argument('--mode', required=True, choices=['live', 'image', 'video'],
                       help='Inference mode: live (camera feed), image (single capture/file), video (video file)')
    parser.add_argument('--model', default='models/best.pt',
                       help='Path to trained model')
    parser.add_argument('--source', type=str, default=None,
                       help='Source file path — image file for "image" mode, video file for "video" mode. '
                            'If not provided in "image" mode, captures from camera.')
    parser.add_argument('--config', default='config/system_config.json',
                       help='Path to system config')
    parser.add_argument('--conf', type=float, default=0.03,
                       help='Confidence threshold (default: 0.3)')
    parser.add_argument('--mock', action='store_true',
                       help='Use mock camera for testing (no hardware needed)')
    
    args = parser.parse_args()
    
    # ---- MODE: VIDEO ----
    if args.mode == 'video':
        if not args.source:
            print("Error: --source is required for video mode.")
            print("Usage: python defect_detector.py --mode video --model best.pt --source video.mp4")
            return
        detector = DefectDetector(args.model, args.config, conf_threshold=args.conf)
        output_path = args.source.replace('.mp4', '_output.mp4')
        print(f"Output will be saved to: {output_path}")
        detector.detect_from_video(args.source, output_path)
        return

    # ---- MODE: IMAGE ----
    if args.mode == 'image':
        if args.source:
            # Detect from image file
            detector = DefectDetector(args.model, args.config, conf_threshold=args.conf)
            result = detector.detect_from_file(args.source)
            
            if result.get('error'):
                print(f"Error: {result['error']}")
                return
            
            # Show annotated image in window
            image = cv2.imread(args.source)
            annotated = detector.draw_results(image, result)
            
            if result['is_defective']:
                print(f"REJECT - {result['defect_count']} defect(s)")
                for d in result['defects']:
                    print(f"  {d['class']}: {d['confidence']:.1%}")
            else:
                print("PASS - No defects")
            print(f"Inference time: {result['inference_time_ms']:.1f}ms")
            
            cv2.namedWindow("Defect Detection - Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Defect Detection - Image", annotated)
            print("\nPress any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            # Capture from camera and show
            system = ProductionInspectionSystem(
                model_path=args.model,
                config_path=args.config,
                conf_threshold=args.conf,
                use_mock_camera=args.mock
            )
            system.camera.connect()
            
            image = system.camera.capture()
            if image is None:
                print("Camera capture failed")
                system.camera.disconnect()
                return
            
            result = system.detector.detect(image)
            annotated = system.detector.draw_results(image, result)
            
            if result['is_defective']:
                print(f"REJECT - {result['defect_count']} defect(s)")
                for d in result['defects']:
                    print(f"  {d['class']}: {d['confidence']:.1%}")
            else:
                print("PASS - No defects")
            print(f"Inference time: {result['inference_time_ms']:.1f}ms")
            
            cv2.namedWindow("Defect Detection - Camera Capture", cv2.WINDOW_NORMAL)
            cv2.imshow("Defect Detection - Camera Capture", annotated)
            print("\nPress 's' to save, any other key to close...")
            key = cv2.waitKey(0) & 0xFF
            if key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"logs/capture_{timestamp}.png"
                os.makedirs("logs", exist_ok=True)
                cv2.imwrite(save_path, annotated)
                print(f"Saved: {save_path}")
            cv2.destroyAllWindows()
            system.camera.disconnect()
        return

    # ---- MODE: LIVE ----
    if args.mode == 'live':
        system = ProductionInspectionSystem(
            model_path=args.model,
            config_path=args.config,
            conf_threshold=args.conf,
            use_mock_camera=args.mock
        )
        system.start()


if __name__ == "__main__":
    main()

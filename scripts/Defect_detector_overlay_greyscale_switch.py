"""
=============================================================
  Defect_detector_overlay_greyscale_switch.py  --  Defect detection pipeline script
=============================================================
HOW TO USE
----------
python Defect_detector_overlay_greyscale_switch.py [-h] --mode

Examples:
python defect_detector.py --mode live  --model best.pt
  python defect_detector.py --mode live  --model best.pt --display grey
  python defect_detector.py --mode image --model best.pt --source photo.jpg --display overlay
  python defect_detector.py --mode video --model best.pt --source vid.mp4   --display grey

FLAGS
-----
-h, --help            show this help message and exit
    --mode {live,image,video}
    Inference mode
    --model MODEL         Path to trained model
    --source SOURCE       Source file â€” image for "image" mode, video for
    "video" mode
    --config CONFIG       Path to system config
    --conf CONF           Confidence threshold (default: 0.03)
    --mock                Use mock camera (no hardware needed)
    --display {overlay,grey}
    What to show in the output window: overlay â€” colour
    frame blended on top of greyscale (default) grey â€”
    exact greyscale frame the model runs on
    Modes:
    live    - Live camera feed (press 'q' quit | 's' snapshot | 'v' toggle display)
    image   - Single capture from camera or image file
    video   - Process a video file (.mp4)
    Display options (--display):
    overlay  - Original colour blended on top of the greyscale frame (default)
    grey     - Pure greyscale frame as seen by the model
    Examples:
    python defect_detector.py --mode live  --model best.pt
    python defect_detector.py --mode live  --model best.pt --display grey
    python defect_detector.py --mode image --model best.pt --source photo.jpg --display overlay
    python defect_detector.py --mode video --model best.pt --source vid.mp4   --display grey
=============================================================
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
    
    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_side_by_side_view(
        ann_original: np.ndarray,
        ann_grey: np.ndarray,
    ) -> np.ndarray:
        """
        Stack an annotated colour frame (left) and an annotated greyscale
        frame (right) into a single side-by-side image with header labels.

        Both panels are the same size as the source frames.
        """
        h, w = ann_original.shape[:2]

        # Header bar per panel (40 px tall)
        bar_h = 40
        bar_orig = np.zeros((bar_h, w, 3), dtype=np.uint8)   # dark bar
        bar_grey = np.zeros((bar_h, w, 3), dtype=np.uint8)

        cv2.putText(bar_orig, "ORIGINAL",
                    (10, bar_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        cv2.putText(bar_grey, "GREYSCALE  (model input)",
                    (10, bar_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 255, 180), 2)

        panel_orig = np.vstack([bar_orig, ann_original])
        panel_grey = np.vstack([bar_grey, ann_grey])

        # 4-px white divider
        divider = np.full((h + bar_h, 4, 3), 220, dtype=np.uint8)

        return np.hstack([panel_orig, divider, panel_grey])

    def detect(self, image: np.ndarray, display_mode: str = 'overlay') -> Dict:
        """
        Run defect detection on image.
        The frame is first converted to rust-aware greyscale (CLAHE + rust
        darkening) — matching the preprocessing used during training —
        before being passed to the YOLO model.

        Args:
            image:        BGR image as numpy array.
            display_mode: 'overlay' or 'grey' — controls what goes into
                          result['display_frame'] for the caller to show.

        Returns:
            dict with detection results, plus 'display_frame' key.
        """
        start_time = time.time()

        # ── Step 1: Resize FIRST to reduce CPU preprocessing workload ──
        # Shrinks from 5MP (2592x1944) → 1280x1280 (~68% fewer pixels)
        # This must happen before greyscale so the heavy math runs on a small image
        image_resized = cv2.resize(image, (1280, 1280))

        # ── Step 2: Greyscale pre-processing on the smaller 1280px image ──
        grey_image = convert_to_greyscale(image_resized)

        # Run inference on greyscale frame at 1280 (matches training resolution)
        results = self.model(
            grey_image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=1280,
            device=self.device,
            verbose=False
        )
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Parse results
        defects = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                defects.append({
                    'class': self.class_names[class_id],
                    'class_id': class_id,
                    'confidence': round(float(box.conf), 4),
                    'bbox': [round(x, 2) for x in box.xyxy.tolist()[0]],
                    'bbox_normalized': [round(x, 4) for x in box.xywhn.tolist()[0]]
                })
        
        # Sort by confidence (highest first)
        defects.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'is_defective': len(defects) > 0,
            'defect_count': len(defects),
            'defects': defects,
            'inference_time_ms': round(inference_time_ms, 2),
            'timestamp': datetime.now().isoformat(),
            'original_frame': image_resized,  # resized colour frame (1280px)
            'grey_frame': grey_image,         # greyscale frame the model saw
        }

    def detect_from_file(self, image_path: str, display_mode: str = 'overlay') -> Dict:
        """Load image from file and run detection (greyscale pre-processing applied automatically)."""
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f'Failed to load image: {image_path}'}

        result = self.detect(image, display_mode=display_mode)
        result['image_path'] = image_path
        return result
    
    def detect_from_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display_mode: str = 'overlay'
    ):
        """Run detection on a video file.

        Args:
            video_path:   Path to input video.
            output_path:  Optional path for annotated output video.
            display_mode: 'overlay' or 'grey' — controls what is written/shown.
        """
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

        print(f"Processing video : {video_path}")
        print(f"Display mode     : {display_mode}")
        print("Press 'q' to quit early.")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.detect(frame)

                if display_mode == 'overlay':
                    ann_orig = self.draw_results(result['original_frame'].copy(), result)
                    ann_grey = self.draw_results(result['grey_frame'].copy(), result)
                    annotated = self.build_side_by_side_view(ann_orig, ann_grey)
                else:
                    annotated = self.draw_results(result['grey_frame'].copy(), result)

                if out is not None:
                    out.write(annotated)

                cv2.imshow("Defect Detection - Video", cv2.resize(annotated, (1200, 600)))
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
            'scratch': (0, 255, 255),      # Yellow
            'crack': (0, 0, 255),          # Red
            'dent': (255, 0, 255),         # Magenta
            'discoloration': (255, 165, 0), # Orange
            'contamination': (0, 255, 0),   # Green
        }
        default_color = (255, 255, 255)
        
        for defect in result.get('defects', []):
            bbox = defect['bbox']
            x1, y1, x2, y2 = [int(x) for x in bbox]
            
            color = colors.get(defect['class'], default_color)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)
            
            # Draw label
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
        # Open one persistent connection for the entire session
        self.conn = sqlite3.connect(self.db_path)
        self._init_db()

    def close(self):
        """Close the persistent database connection at end of session"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def _init_db(self):
        """Create database schema"""
        self.conn.execute('''
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
        self.conn.commit()
    
    def log(self, result: Dict, image_path: str = None,
            shift: str = None, batch: str = None):
        """Log inspection result using the persistent connection"""
        max_conf = max([d['confidence'] for d in result['defects']]) if result['defects'] else 0

        self.conn.execute('''
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
        self.conn.commit()
    
    def get_stats(self, hours: int = 24) -> Dict:
        """Get inspection statistics for the last N hours"""
        # Use the persistent connection — no open/close needed
        results = self.conn.execute('''
            SELECT result, COUNT(*) FROM inspections
            WHERE datetime(timestamp) > datetime('now', ?)
            GROUP BY result
        ''', (f'-{hours} hours',)).fetchall()

        stats = {result: count for result, count in results}
        total = sum(stats.values())

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
    
    def start(self, display_mode: str = 'overlay'):
        """
        Start continuous inspection mode with live video feed.

        Args:
            display_mode: 'overlay' — colour blended on top of greyscale
                          'grey'    — exact greyscale frame the model sees
                          Press 'v' during live feed to toggle between modes.
        """
        if not self.camera.connect():
            print("Failed to connect to camera")
            return

        self._running = True
        current_mode = display_mode

        mode_label = {'overlay': 'OVERLAY (colour + grey)', 'grey': 'GREYSCALE only'}
        print("\n" + "="*55)
        print("LIVE INSPECTION MODE")
        print(f"Display : {mode_label.get(current_mode, current_mode)}")
        print("Keys    : 'q' quit  |  's' snapshot  |  'v' toggle display")
        print("="*55 + "\n")

        frame_count = 0
        fps_start = time.time()
        display_fps = 0.0

        try:
            while self._running:
                # Capture frame
                image = self.camera.capture()
                if image is None:
                    print("Camera capture failed, retrying...")
                    time.sleep(0.5)
                    continue

                # Run detection (greyscale pre-proc inside)
                result = self.detector.detect(image)

                # Build the display frame based on chosen mode
                if current_mode == 'overlay':
                    ann_orig = self.detector.draw_results(result['original_frame'].copy(), result)
                    ann_grey = self.detector.draw_results(result['grey_frame'].copy(), result)
                    annotated = self.detector.build_side_by_side_view(ann_orig, ann_grey)
                else:
                    annotated = self.detector.draw_results(result['grey_frame'].copy(), result)

                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    display_fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.time()

                # HUD overlay
                h = annotated.shape[0]
                cv2.putText(annotated, f"FPS: {display_fps:.1f}",
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated, f"Defects: {result['defect_count']}",
                            (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated, f"[v] {mode_label.get(current_mode, current_mode)}",
                            (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                # Show live feed
                cv2.namedWindow("Defect Detection - Live Feed", cv2.WINDOW_NORMAL)
                cv2.imshow("Defect Detection - Live Feed", annotated)

                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting live feed...")
                    break
                elif key == ord('v'):
                    # Toggle display mode
                    current_mode = 'grey' if current_mode == 'overlay' else 'overlay'
                    print(f"  Display → {mode_label[current_mode]}")
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    status = 'reject' if result['is_defective'] else 'pass'
                    save_dir = Path("logs/inspections") / status
                    save_dir.mkdir(parents=True, exist_ok=True)
                    snap_path = save_dir / f"snapshot_{status}_{timestamp}.png"
                    cv2.imwrite(str(snap_path), annotated)
                    print(f"  Snapshot saved: {snap_path}")

                # Log to database
                self.logger.log(result)

        except KeyboardInterrupt:
            print("\nStopping inspection...")
        finally:
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

        # Close the persistent database connection
        self.logger.close()


def main():
    """Main entry point for production inference"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Defect Detection Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  live    - Live camera feed (press 'q' quit | 's' snapshot | 'v' toggle display)
  image   - Single capture from camera or image file
  video   - Process a video file (.mp4)

Display options (--display):
  overlay  - Original colour blended on top of the greyscale frame (default)
  grey     - Pure greyscale frame as seen by the model

Examples:
  python defect_detector.py --mode live  --model best.pt
  python defect_detector.py --mode live  --model best.pt --display grey
  python defect_detector.py --mode image --model best.pt --source photo.jpg --display overlay
  python defect_detector.py --mode video --model best.pt --source vid.mp4   --display grey
        """
    )
    parser.add_argument('--mode', required=True, choices=['live', 'image', 'video'],
                        help='Inference mode')
    parser.add_argument('--model', default='models/best.pt',
                        help='Path to trained model')
    parser.add_argument('--source', type=str, default=None,
                        help='Source file — image for "image" mode, video for "video" mode')
    parser.add_argument('--config', default='config/system_config.json',
                        help='Path to system config')
    parser.add_argument('--conf', type=float, default=0.03,
                        help='Confidence threshold (default: 0.03)')
    parser.add_argument('--mock', action='store_true',
                        help='Use mock camera (no hardware needed)')
    parser.add_argument(
        '--display',
        choices=['overlay', 'grey'],
        default='overlay',
        help=(
            'What to show in the output window:\n'
            '  overlay — colour frame blended on top of greyscale (default)\n'
            '  grey    — exact greyscale frame the model runs on'
        )
    )

    args = parser.parse_args()

    # ---- MODE: VIDEO ----
    if args.mode == 'video':
        if not args.source:
            print("Error: --source is required for video mode.")
            print("Usage: python defect_detector.py --mode video --model best.pt --source video.mp4")
            return
        detector = DefectDetector(args.model, args.config, conf_threshold=args.conf)
        output_path = args.source.replace('.mp4', '_output.mp4')
        print(f"Output will be saved to: {output_path}  (display: {args.display})")
        detector.detect_from_video(args.source, output_path, display_mode=args.display)
        return

    # ---- MODE: IMAGE ----
    if args.mode == 'image':
        if args.source:
            # Detect from image file
            detector = DefectDetector(args.model, args.config, conf_threshold=args.conf)
            result = detector.detect_from_file(args.source, display_mode=args.display)

            if result.get('error'):
                print(f"Error: {result['error']}")
                return

            # Build the annotated display
            if args.display == 'overlay':
                ann_orig = detector.draw_results(result['original_frame'].copy(), result)
                ann_grey = detector.draw_results(result['grey_frame'].copy(), result)
                annotated = detector.build_side_by_side_view(ann_orig, ann_grey)
            else:
                annotated = detector.draw_results(result['grey_frame'].copy(), result)

            if result['is_defective']:
                print(f"REJECT - {result['defect_count']} defect(s)")
                for d in result['defects']:
                    print(f"  {d['class']}: {d['confidence']:.1%}")
            else:
                print("PASS - No defects")
            print(f"Inference time: {result['inference_time_ms']:.1f}ms")
            print(f"Display mode  : {args.display}")

            cv2.namedWindow("Defect Detection - Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Defect Detection - Image", annotated)
            print("\nPress any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            # Capture from camera
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
            if args.display == 'overlay':
                ann_orig = system.detector.draw_results(result['original_frame'].copy(), result)
                ann_grey = system.detector.draw_results(result['grey_frame'].copy(), result)
                annotated = system.detector.build_side_by_side_view(ann_orig, ann_grey)
            else:
                annotated = system.detector.draw_results(result['grey_frame'].copy(), result)

            if result['is_defective']:
                print(f"REJECT - {result['defect_count']} defect(s)")
                for d in result['defects']:
                    print(f"  {d['class']}: {d['confidence']:.1%}")
            else:
                print("PASS - No defects")
            print(f"Inference time: {result['inference_time_ms']:.1f}ms")
            print(f"Display mode  : {args.display}")

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
        system.start(display_mode=args.display)


if __name__ == "__main__":
    main()

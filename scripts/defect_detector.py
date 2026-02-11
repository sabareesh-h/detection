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
from camera_capture import get_camera, BaslerCamera, MockCamera


class DefectDetector:
    """Production defect detection system"""
    
    def __init__(
        self,
        model_path: str,
        config_path: str = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
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
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45,
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
        Run defect detection on image
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            dict with detection results
        """
        start_time = time.time()
        
        # Run inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
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
            'timestamp': datetime.now().isoformat()
        }
    
    def detect_from_file(self, image_path: str) -> Dict:
        """Load image from file and run detection"""
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f'Failed to load image: {image_path}'}
        
        result = self.detect(image)
        result['image_path'] = image_path
        return result
    
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
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{defect['class']}: {defect['confidence']:.1%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw overall status
        status = "REJECT" if result['is_defective'] else "PASS"
        status_color = (0, 0, 255) if result['is_defective'] else (0, 255, 0)
        cv2.putText(annotated, status, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
        
        # Draw inference time
        cv2.putText(annotated, f"{result['inference_time_ms']:.1f}ms", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
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
        use_mock_camera: bool = False
    ):
        """
        Initialize production inspection system
        
        Args:
            model_path: Path to trained model
            config_path: Path to system configuration
            use_mock_camera: Use mock camera for testing
        """
        self.detector = DefectDetector(model_path, config_path)
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
        """Start continuous inspection mode"""
        if not self.camera.connect():
            print("Failed to connect to camera")
            return
        
        self._running = True
        print("\n" + "="*50)
        print("CONTINUOUS INSPECTION MODE")
        print("Press Ctrl+C to stop")
        print("="*50 + "\n")
        
        try:
            while self._running:
                result = self.inspect_once()
                self.print_result(result)
                time.sleep(0.1)  # Small delay between inspections
                
        except KeyboardInterrupt:
            print("\nStopping inspection...")
        finally:
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
    
    parser = argparse.ArgumentParser(description='Defect Detection Inference')
    parser.add_argument('--model', default='models/best.pt',
                       help='Path to trained model')
    parser.add_argument('--config', default='config/system_config.json',
                       help='Path to system config')
    parser.add_argument('--image', type=str, default=None,
                       help='Single image to inspect (instead of camera)')
    parser.add_argument('--mock', action='store_true',
                       help='Use mock camera for testing')
    parser.add_argument('--continuous', action='store_true',
                       help='Continuous inspection mode')
    
    args = parser.parse_args()
    
    # Single image mode
    if args.image:
        detector = DefectDetector(args.model, args.config)
        result = detector.detect_from_file(args.image)
        
        if result.get('error'):
            print(f"Error: {result['error']}")
        elif result['is_defective']:
            print(f"REJECT - {result['defect_count']} defect(s)")
            for d in result['defects']:
                print(f"  {d['class']}: {d['confidence']:.1%}")
        else:
            print("PASS - No defects")
        
        print(f"Inference time: {result['inference_time_ms']:.1f}ms")
        return
    
    # Camera mode
    system = ProductionInspectionSystem(
        model_path=args.model,
        config_path=args.config,
        use_mock_camera=args.mock
    )
    
    if args.continuous:
        system.start()
    else:
        # Single inspection
        system.camera.connect()
        result = system.inspect_once()
        system.print_result(result)
        system.camera.disconnect()


if __name__ == "__main__":
    main()

r"""
=============================================================
  defect_detector.py  --  Defect detection pipeline script
=============================================================
HOW TO USE
----------
python defect_detector.py [-h] --mode {live,image,video} [--model MODEL]

Examples:
python defect_detector.py --mode live --model path/to/best.pt
  python defect_detector.py --mode image --model path/to/best.pt
  python defect_detector.py --mode video --model path/to/best.pt --source video.mp4
  python defect_detector.py --mode image --model path/to/best.pt --source photo.jpg

FLAGS
-----
-h, --help            show this help message and exit
    --mode {live,image,video}
    Inference mode: live (camera feed), image (single
    capture/file), video (video file)
    --model MODEL         Path to trained model
    --source SOURCE       Source file path â€” image file for "image" mode, video
    file for "video" mode. If not provided in "image"
    mode, captures from camera.
    --config CONFIG       Path to system config
    --conf CONF           Confidence threshold (default: 0.3)
    --mock                Use mock camera for testing (no hardware needed)
    Modes:
    live    - Live camera feed with bounding boxes (press 'q' to quit, 's' to snapshot)
    image   - Capture single image from camera, show detection result
    video   - Process a video file (.mp4) with detection
    Examples:
    python defect_detector.py --mode live --model path/to/best.pt
    python defect_detector.py --mode image --model path/to/best.pt
    python defect_detector.py --mode video --model path/to/best.pt --source video.mp4
    python defect_detector.py --mode image --model path/to/best.pt --source photo.jpg

OLD EXAMPLES / SETUP
--------------------
# scripts to run
# Opening Environment   - .\defect_env_gpu311\Scripts\activate
# Directing to scripts - cd scripts
# Detecting defects - python defect_detector.py --mode live --model runs/detect/my_first_wandb_run/weights/best.pt --config config/system_config.json --conf 0.03
# Detecting Defect with greyscale - python defect_detector_overlay_greyscale_switch.py --mode live --model runs/detect/models/my_first_wandb_run6/weights/best.pt --display grey
# Detecting live - python defect_detector.py --mode live --model runs/detect/models/my_first_wandb_run6/weights/best.pt 
# Detecting Image - python defect_detector.py --mode image --model runs/detect/models/my_first_wandb_run6/weights/best.pt --source C:\Users\RohithSuryaCKM\Downloads\Projects\Image_detection\scripts\test_images\test_image_1.jpg
# detecting video - python defect_detector.py --mode video --model runs/detect/models/my_first_wandb_run6/weights/best.pt --source C:\Users\RohithSuryaCKM\Downloads\Projects\Image_detection\scripts\test_images\test_video_1.mp4
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
    from ultralytics import YOLO, RTDETR
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from config_manager import load_system_config



def _load_model(model_path: str):
    """Auto-detect YOLO vs RT-DETR from filename and return the right model instance."""
    if 'rtdetr' in str(model_path).lower():
        print(f"[Model] Detected RT-DETR weights — loading with RTDETR class.")
        return RTDETR(model_path)
    return YOLO(model_path)

# Import camera module
from camera_capture import get_camera, BaslerCamera, MockCamera, convert_to_greyscale, convert_to_greyscale_gpu


class DefectDetector:
    """Production defect detection system"""
    
    # Zone names in order (left → right on the part)
    ZONE_NAMES = ["Top", "Mid", "Thread", "Bottom"]

    def __init__(
        self,
        model_path: str,
        config_path: str = None,
        conf_threshold: float = 0.03,
        iou_threshold: float = 0.45,
        device: str = '0'
    ):
        """
        Initialize defect detector
        
        Args:
            model_path: Path to trained YOLOv11 model weights
            config_path: Path to system_config.json
            conf_threshold: Global minimum confidence (used as the default for all 4 zones)
            iou_threshold: IoU threshold for NMS (higher = fewer overlapping duplicates suppressed;
                           0.45 is the recommended default for segmentation models)
            device: GPU device ('0', '1', 'cpu')
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics not installed")
        
        self.model_path = model_path
        # Global model-level threshold — kept low so the model sees everything.
        # Per-zone thresholds (zone_conf) are applied AFTER inference.
        self.conf_threshold = 0.01
        # Per-zone confidence thresholds: [Top, Mid, Thread, Bottom]
        self.zone_conf = [conf_threshold] * 4
        self.iou_threshold = iou_threshold  # Raised to 0.45 to suppress overlapping duplicate masks
        self.device = device
        
        # Load configuration
        self.config = load_system_config(config_path)
        
        # Min Area px size rejection limit (configurable from UI)
        self.min_area_px = self.config.inspection.min_defect_area_px
        
        # Load model
        print(f"Loading model: {model_path}")
        self.model = _load_model(model_path)
        self.class_names = self.model.names

        # Auto-detect whether this is a detection or segmentation model
        self.is_seg = getattr(self.model, 'task', 'detect') == 'segment'
        print(f"Model type: {'SEGMENTATION (mask)' if self.is_seg else 'DETECTION (bbox)'}")

        # Warmup model
        self._warmup()

        print(f"Detector initialized. Classes: {list(self.class_names.values())}")
    

    
    def _warmup(self, iterations: int = 3):
        """Warmup model with dummy inference"""
        print("Warming up model...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.model(dummy, verbose=False)
        print("Warmup complete")

    # ──────────────────────────────────────────────────────────────
    #  Post-processing helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _mask_iou(poly_a: list, poly_b: list, img_shape: tuple) -> float:
        """
        Compute pixel-level IoU between two mask polygons.
        Optimized to only create masks within their bounding box union.
        """
        pts_a = np.array(poly_a, dtype=np.int32)
        pts_b = np.array(poly_b, dtype=np.int32)
        
        # Bounding boxes
        xa_min, ya_min = pts_a.min(axis=0)
        xa_max, ya_max = pts_a.max(axis=0)
        xb_min, yb_min = pts_b.min(axis=0)
        xb_max, yb_max = pts_b.max(axis=0)
        
        # Check bbox intersection
        inter_xmin = max(xa_min, xb_min)
        inter_ymin = max(ya_min, yb_min)
        inter_xmax = min(xa_max, xb_max)
        inter_ymax = min(ya_max, yb_max)
        
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
            
        # Optimization: use bounding box of union
        w_roi = max(xa_max, xb_max) - min(xa_min, xb_min) + 1
        h_roi = max(ya_max, yb_max) - min(ya_min, yb_min) + 1
        x_min = min(xa_min, xb_min)
        y_min = min(ya_min, yb_min)
        
        mask_a = np.zeros((h_roi, w_roi), dtype=np.uint8)
        mask_b = np.zeros((h_roi, w_roi), dtype=np.uint8)
        
        pts_a_shifted = pts_a - [x_min, y_min]
        pts_b_shifted = pts_b - [x_min, y_min]
        
        cv2.fillPoly(mask_a, [pts_a_shifted], 1)
        cv2.fillPoly(mask_b, [pts_b_shifted], 1)
        
        intersection = np.logical_and(mask_a, mask_b).sum()
        if intersection == 0:
            return 0.0
        union = np.logical_or(mask_a, mask_b).sum()
        return float(intersection) / float(union) if union > 0 else 0.0

    def _deduplicate_by_mask_iou(
        self, defects: list, img_shape: tuple, iou_thresh: float = 0.35
    ) -> list:
        """
        Second-pass NMS using actual mask pixels instead of bounding boxes.
        Suppresses lower-confidence detections whose mask overlaps a
        higher-confidence one by more than `iou_thresh`.
        Defects must be sorted by confidence (highest first) on entry.
        """
        suppressed: set = set()
        for i, d_i in enumerate(defects):
            if i in suppressed:
                continue
            poly_i = d_i.get('mask_polygon')
            if not poly_i or len(poly_i) < 3:
                continue
            for j in range(i + 1, len(defects)):
                if j in suppressed:
                    continue
                poly_j = defects[j].get('mask_polygon')
                if not poly_j or len(poly_j) < 3:
                    continue
                if self._mask_iou(poly_i, poly_j, img_shape) > iou_thresh:
                    suppressed.add(j)
        return [d for idx, d in enumerate(defects) if idx not in suppressed]

    @staticmethod
    def _cluster_defects(defects: list, distance_thresh: float = 100.0) -> list:
        """
        Merge detections whose mask centroids are within `distance_thresh`
        pixels of each other — handles cases where one long scratch is
        detected as two separate fragments with no pixel overlap.
        The highest-confidence detection in each cluster is kept as the
        representative; its bounding box is expanded to cover all members.
        """
        if not defects:
            return defects

        def _centroid(d):
            poly = d.get('mask_polygon')
            if poly and len(poly) >= 3:
                pts = np.array(poly)
                return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            x1, y1, x2, y2 = d['bbox']
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0

        centroids = [_centroid(d) for d in defects]
        visited   = [False] * len(defects)
        merged    = []

        for i in range(len(defects)):
            if visited[i]:
                continue
            cluster_ids = [i]
            visited[i]  = True
            for j in range(i + 1, len(defects)):
                if visited[j]:
                    continue
                dx = centroids[i][0] - centroids[j][0]
                dy = centroids[i][1] - centroids[j][1]
                if (dx * dx + dy * dy) ** 0.5 < distance_thresh:
                    cluster_ids.append(j)
                    visited[j] = True

            members = [defects[k] for k in cluster_ids]
            best    = max(members, key=lambda d: d['confidence'])

            if len(members) > 1:
                best = dict(best)  # copy so we don't mutate original
                best['bbox'] = [
                    min(d['bbox'][0] for d in members),
                    min(d['bbox'][1] for d in members),
                    max(d['bbox'][2] for d in members),
                    max(d['bbox'][3] for d in members),
                ]
                best['merged_count'] = len(members)

            merged.append(best)

        return merged

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
        
        # ── ROI Cropping ──
        roi = self.config.inspection.roi
        h, w = image.shape[:2]
        offset_x, offset_y = 0, 0
        crop_image = image
        
        if roi and len(roi) == 4:
            ymin, ymax, xmin, xmax = roi
            ymin, ymax = max(0, ymin), min(h, ymax)
            xmin, xmax = max(0, xmin), min(w, xmax)
            if ymax > ymin and xmax > xmin:
                crop_image = image[ymin:ymax, xmin:xmax]
                offset_x, offset_y = xmin, ymin
                
        # ── Greyscale pre-processing ──
        if self.config.model.gpu_preprocessing:
            grey_image = convert_to_greyscale_gpu(crop_image, device=self.device)
        else:
            grey_image = convert_to_greyscale(crop_image)
        
        # ── Idea 1: Dynamic Horizontal Band Focus ──
        masked_image = grey_image.copy()
        core_box = None
        dynamic_margin = self.config.inspection.dynamic_margin
        
        if dynamic_margin > 0:
            blurred = cv2.GaussianBlur(grey_image, (7, 7), 0)
            edges = cv2.Canny(blurred, 40, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(main_contour) > 5000:
                    x, y, w_c, h_c = cv2.boundingRect(main_contour)
                    
                    # Apply margin to Y extent only
                    margin_y = int(h_c * dynamic_margin)
                    y_start = max(0, y + margin_y)
                    y_end = min(grey_image.shape[0], y + h_c - margin_y)
                    
                    if y_end > y_start:
                        core_box = np.array([[0, y_start], [grey_image.shape[1], y_start], 
                                             [grey_image.shape[1], y_end], [0, y_end]], dtype=np.int32)
                        
                        mask = np.zeros(grey_image.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask, [core_box], 255)
                        masked_image = cv2.bitwise_and(grey_image, grey_image, mask=mask)
                        
                        # Shift dynamic box to original image coordinates for drawing
                        core_box[:, 0] += offset_x
                        core_box[:, 1] += offset_y

        # Run inference on completely shielded dynamic spotlight.
        # Use a very low global threshold so the model returns everything;
        # per-zone thresholds are applied below in post-processing.
        results = self.model(
            masked_image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,       # NMS IoU threshold (0.45 = suppress >45% overlapping duplicates)
            agnostic_nms=False,            # Keep class-aware NMS (all defects are same class anyway)
            device=self.device,
            verbose=False
        )
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Parse results
        all_defects = []
        for result in results:
            masks = result.masks  # None for detection models
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls)
                
                # shift bbox back to full image scaling
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                x1 += offset_x; y1 += offset_y
                x2 += offset_x; y2 += offset_y
                bbox_orig = [x1, y1, x2, y2]
                defect = {
                    'class':          self.class_names[class_id],
                    'class_id':       class_id,
                    'confidence':     round(float(box.conf), 4),
                    'bbox':           [round(x, 2) for x in bbox_orig],
                    'bbox_normalized':[round(x1/w, 4), round(y1/h, 4), round((x2-x1)/w, 4), round((y2-y1)/h, 4)],
                    'mask_polygon':   None,
                    'area_px':        0.0
                }
                
                # Extract mask polygon and calculate true area
                area_px = 0.0
                if masks is not None and i < len(masks):
                    try:
                        xy = masks.xy[i]          # (N, 2) array of pixel coords
                        xy[:, 0] += offset_x      # shift polygon coordinates to original
                        xy[:, 1] += offset_y
                        defect['mask_polygon'] = xy.tolist()
                        
                        contour = np.array(xy, dtype=np.float32).reshape((-1, 1, 2))
                        area_px = cv2.contourArea(contour)
                    except Exception as e:
                        print(f"Error calculating contour area: {e}")
                        area_px = float((x2 - x1) * (y2 - y1))
                else:
                    area_px = float((x2 - x1) * (y2 - y1))
                    
                defect['area_px'] = round(area_px, 2)
                
                # ==== Size Rejection Filtering ====
                if area_px < self.min_area_px:
                    continue  # Ignore defects smaller than this area threshold
                    
                all_defects.append(defect)
        
        # ── Per-zone confidence filtering ──
        # Assign each detected defect to a zone by its centroid X,
        # then keep only those whose confidence meets THAT zone's threshold.
        num_zones = 4
        zone_width_px = w // num_zones
        defects = []
        for defect in all_defects:
            mask_poly = defect.get('mask_polygon')
            if mask_poly and len(mask_poly) >= 3:
                pts_arr = np.array(mask_poly)
                cx = float(np.mean(pts_arr[:, 0]))
            else:
                bx1, _, bx2, _ = defect['bbox']
                cx = (bx1 + bx2) / 2
            zone_idx = min(int(cx) // zone_width_px, num_zones - 1)
            defect['zone_idx'] = zone_idx  # store for draw_results
            if defect['confidence'] >= self.zone_conf[zone_idx]:
                defects.append(defect)

        # Sort by confidence (highest first)
        defects.sort(key=lambda x: x['confidence'], reverse=True)

        # ── Pass 1: pixel-level mask IoU deduplication ──
        # Catches overlapping masks that bounding-box NMS misses.
        defects = self._deduplicate_by_mask_iou(
            defects, grey_image.shape, iou_thresh=0.35
        )

        # ── Pass 2: centroid-distance clustering ──
        # Merges fragments of the same physical scratch that don't overlap.
        defects = self._cluster_defects(defects, distance_thresh=100.0)

        # Re-sort after merging (cluster representative keeps best conf)
        defects.sort(key=lambda x: x['confidence'], reverse=True)

        return {
            'is_defective': len(defects) > 0,
            'defect_count': len(defects),
            'defects': defects,
            'inference_time_ms': round(inference_time_ms, 2),
            'timestamp': datetime.now().isoformat(),
            'dynamic_core_box': core_box.tolist() if core_box is not None else None
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
        
        window_name = "Defect Detection - Video"
        settings_name = "Settings"
        show_settings = False
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if show_settings:
                    try:
                        if cv2.getWindowProperty(settings_name, cv2.WND_PROP_VISIBLE) < 1:
                            show_settings = False
                        else:
                            for zi, zname in enumerate(self.ZONE_NAMES):
                                pct = cv2.getTrackbarPos(f"Conf {zname} %", settings_name)
                                if pct is not None:
                                    self.zone_conf[zi] = max(1, pct) / 100.0
                            
                            area_val = cv2.getTrackbarPos("Min Area px", settings_name)
                            if area_val is not None:
                                self.min_area_px = float(area_val)
                            
                            gpu_val = cv2.getTrackbarPos("GPU Preprocess", settings_name)
                            if gpu_val is not None:
                                self.config.model.gpu_preprocessing = bool(gpu_val)
                    except cv2.error:
                        show_settings = False
                
                # Convert to greyscale before inference; draw results on original frame
                result = self.detect(frame)          # greyscale applied inside detect()
                annotated = self.draw_results(frame, result)
                
                if out is not None:
                    out.write(annotated)
                
                cv2.putText(annotated, "[T] TOGGLE CONFIG SLIDERS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.imshow(window_name, cv2.resize(annotated, (800, 600)))
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t') or key == 80:
                    show_settings = not show_settings
                    if show_settings:
                        cv2.namedWindow(settings_name, cv2.WINDOW_NORMAL)
                        for zi, zname in enumerate(self.ZONE_NAMES):
                            cv2.createTrackbar(f"Conf {zname} %", settings_name, int(self.zone_conf[zi] * 100), 100, lambda x: None)
                        cv2.createTrackbar("Min Area px", settings_name, int(self.min_area_px), 5000, lambda x: None)
                        cv2.createTrackbar("GPU Preprocess", settings_name, int(self.config.model.gpu_preprocessing), 1, lambda x: None)
                    else:
                        try: cv2.destroyWindow(settings_name)
                        except Exception as e: print(f"Error: {e}")
        finally:
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
    
    def draw_results(self, image: np.ndarray, result: Dict,
                     hover_pos: Optional[Tuple[int, int]] = None,
                     always_show_labels: bool = False) -> np.ndarray:
        """
        Draw detection results on image

        Args:
            image: Original image
            result: Detection result dict
            hover_pos: (x, y) mouse coordinates for hover tooltips (cv2 live mode)
            always_show_labels: If True, always draw confidence labels at mask
                                centroids regardless of hover (used by web server)
            
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
        seg_thickness = max(1, int(1 * scale))  # Thinner segmentation border
        label_pad = int(10 * scale)
        
        colors = {
            's': (0, 0, 255),   # Red (OpenCV uses BGR)
        }
        default_color = (0, 255, 255)
        
        # Draw ROI boundary if active
        roi = self.config.inspection.roi
        if roi and len(roi) == 4:
            ymin, ymax, xmin, xmax = roi
            ymin, ymax = max(0, ymin), min(h, ymax)
            xmin, xmax = max(0, xmin), min(w, xmax)
            # Draw thin white framing box to indicate the actual inspection area
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (255, 255, 255), max(1, int(1 * scale)))
            
        # Draw dynamic core focus area
        dynamic_core = result.get('dynamic_core_box')
        if dynamic_core:
            pts = np.array(dynamic_core, dtype=np.int32)
            # Blue line identifying the exact cut-off area (like your drawing)
            cv2.polylines(annotated, [pts], isClosed=True, color=(255, 150, 0), thickness=max(2, int(2 * scale)))
        
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
                cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=seg_thickness)
            else:
                # ── Detection model: draw plain bounding box ──
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

            # Draw label: always (web mode) or on hover (cv2 live mode)
            show_label = False
            lx, ly = x1, y1   # default label anchor

            if always_show_labels:
                # Anchor label at mask centroid
                if mask_poly and len(mask_poly) >= 3:
                    pts_c = np.array(mask_poly)
                    lx = int(np.mean(pts_c[:, 0]))
                    ly = int(np.mean(pts_c[:, 1]))
                show_label = True
            elif hover_pos is not None:
                hx, hy = hover_pos
                is_hovered = False
                if mask_poly and len(mask_poly) >= 3:
                    if cv2.pointPolygonTest(pts, (hx, hy), False) >= 0:
                        is_hovered = True
                else:
                    if x1 <= hx <= x2 and y1 <= hy <= y2:
                        is_hovered = True
                if is_hovered:
                    lx, ly = hx, hy
                    show_label = True

            if show_label:
                label = f"{defect['class'].upper()}: {defect['confidence']:.1%}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                # keep label inside frame
                lx = min(lx, w - label_size[0] - 6)
                ly = max(ly, label_size[1] + label_pad)
                cv2.rectangle(annotated,
                              (lx, ly - label_size[1] - label_pad),
                              (lx + label_size[0] + 4, ly),
                              color, -1)
                cv2.putText(annotated, label, (lx + 2, ly - int(label_pad / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        # ── Defect index badges (drawn on top of all masks) ── [DISABLED]
        # To re-enable, remove the leading "# " from each line below.
        # badge_r   = max(11, int(13 * scale))
        # badge_fs  = 0.38 * scale
        # badge_thk = max(1, int(2 * scale))
        # for rank, defect in enumerate(result.get('defects', []), start=1):
        #     mask_poly = defect.get('mask_polygon')
        #     if mask_poly and len(mask_poly) >= 3:
        #         pts_b = np.array(mask_poly)
        #         cx_b  = int(np.mean(pts_b[:, 0]))
        #         cy_b  = int(np.mean(pts_b[:, 1]))
        #     else:
        #         bx1, by1, bx2, by2 = [int(x) for x in defect['bbox']]
        #         cx_b = (bx1 + bx2) // 2
        #         cy_b = (by1 + by2) // 2
        #     b_color = colors.get(defect['class'].lower(), default_color)
        #     cv2.circle(annotated, (cx_b, cy_b), badge_r, b_color, -1)
        #     cv2.circle(annotated, (cx_b, cy_b), badge_r, (255, 255, 255),
        #                max(1, int(scale)))
        #     num_str = str(rank)
        #     (nw, nh), _ = cv2.getTextSize(
        #         num_str, cv2.FONT_HERSHEY_SIMPLEX, badge_fs, badge_thk)
        #     cv2.putText(annotated, num_str,
        #                 (cx_b - nw // 2, cy_b + nh // 2),
        #                 cv2.FONT_HERSHEY_SIMPLEX, badge_fs,
        #                 (0, 0, 0), badge_thk)

        # ── Zone Analysis: divide part into 4 vertical zones ──
        num_zones = 4
        zone_width = w // num_zones
        
        # Determine Y extent for the zone divider lines
        dynamic_core = result.get('dynamic_core_box')
        if dynamic_core:
            core_pts = np.array(dynamic_core, dtype=np.int32)
            zone_ymin = int(np.min(core_pts[:, 1]))
            zone_ymax = int(np.max(core_pts[:, 1]))
        else:
            zone_ymin, zone_ymax = 0, h
        
        # Assign each defect to a zone using the zone_idx already computed in detect()
        zone_defects = {zi: [] for zi in range(num_zones)}
        for defect in result.get('defects', []):
            zi = defect.get('zone_idx')
            if zi is None:
                # fallback: compute from centroid (e.g. results from older code)
                mask_poly = defect.get('mask_polygon')
                if mask_poly and len(mask_poly) >= 3:
                    pts_arr = np.array(mask_poly)
                    cx = float(np.mean(pts_arr[:, 0]))
                else:
                    bx1, _, bx2, _ = defect['bbox']
                    cx = (bx1 + bx2) / 2
                zi = min(int(cx) // zone_width, num_zones - 1)
            zone_defects[zi].append(defect)
        
        # Sort each zone's defects by confidence desc
        for zi in zone_defects:
            zone_defects[zi].sort(key=lambda d: d['confidence'], reverse=True)
        
        bad_zones = {zi for zi, defs in zone_defects.items() if len(defs) > 0}
        
        # Drawing constants
        zone_line_color  = (255, 150, 0)
        zone_line_thick  = max(1, int(2 * scale))
        label_font_scale = 0.7 * scale
        label_thick      = max(1, int(2 * scale))
        tbl_font_scale   = 0.45 * scale
        tbl_thick        = max(1, int(1 * scale))
        tbl_color        = (255, 150, 0)   # matches zone lines
        tbl_text_color   = (220, 220, 220)
        
        # Row height for the mini table
        (_, row_h), _ = cv2.getTextSize("Scratch", cv2.FONT_HERSHEY_SIMPLEX, tbl_font_scale, tbl_thick)
        row_h = int(row_h * 2.0)   # a bit of padding

        # Y position of Good/Bad label: anchor bottom of the table slightly above the ROI threshold
        (_, lh_ref), _ = cv2.getTextSize("Bad", cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thick)
        ideal_label_y = zone_ymin - (2 * row_h) - int(12 * scale)
        label_y = max(ideal_label_y, lh_ref + int(10 * scale))
        
        for zi in range(num_zones):
            x_left   = zi * zone_width
            x_right  = (zi + 1) * zone_width if zi < num_zones - 1 else w
            x_center = (x_left + x_right) // 2
            pad      = max(4, int(4 * scale))
            
            # Vertical divider lines (skip leftmost edge)
            if zi > 0:
                cv2.line(annotated, (x_left, zone_ymin), (x_left, zone_ymax),
                         zone_line_color, zone_line_thick)
            
            is_bad = zi in bad_zones
            label  = "Bad" if is_bad else "Good"
            lcolor = (0, 0, 255) if is_bad else (0, 255, 0)

            # --- Zone name header (e.g. "Top") ---
            zone_name = self.ZONE_NAMES[zi] if zi < len(self.ZONE_NAMES) else f"Z{zi}"
            zone_conf_val = self.zone_conf[zi] if zi < len(self.zone_conf) else 0.0
            header_str = f"{zone_name}  [{zone_conf_val:.0%}]"
            header_font = 0.55 * scale
            header_thick = max(1, int(1 * scale))
            (hw, hh), _ = cv2.getTextSize(header_str, cv2.FONT_HERSHEY_SIMPLEX, header_font, header_thick)
            hx = max(x_left + pad, x_center - hw // 2)
            header_y = label_y - lh_ref - int(6 * scale)   # just above the Good/Bad label
            cv2.putText(annotated, header_str, (hx, header_y),
                        cv2.FONT_HERSHEY_SIMPLEX, header_font, (200, 200, 200), header_thick)

            # Center the Good/Bad label text
            (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thick)
            lx = max(x_left + pad, x_center - lw // 2)
            cv2.putText(annotated, label, (lx, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, lcolor, label_thick)
            
            # ── Mini table: top 2 defects ──
            n_rows   = 2
            top_defs = zone_defects[zi][:n_rows]   # may be 0, 1 or 2 items
            
            tbl_x1 = x_left  + pad
            tbl_x2 = x_right - pad
            tbl_y1 = label_y + int(4 * scale)      # just below the label
            tbl_y2 = tbl_y1 + n_rows * row_h
            tbl_mid = (tbl_x1 + tbl_x2) // 2      # vertical divider between name / conf cols
            
            # Outer rectangle
            cv2.rectangle(annotated, (tbl_x1, tbl_y1), (tbl_x2, tbl_y2), tbl_color, tbl_thick)
            # Vertical divider (name | confidence)
            cv2.line(annotated, (tbl_mid, tbl_y1), (tbl_mid, tbl_y2), tbl_color, tbl_thick)
            
            for row in range(n_rows):
                row_y1 = tbl_y1 + row * row_h
                row_y2 = row_y1 + row_h
                # Horizontal row divider (not needed for last row)
                if row > 0:
                    cv2.line(annotated, (tbl_x1, row_y1), (tbl_x2, row_y1), tbl_color, tbl_thick)
                
                # Fill text if defect exists for this row
                if row < len(top_defs):
                    d = top_defs[row]
                    # Class name (capitalise) + rank number
                    name_str = f"{d['class'].upper()} {row + 1}"
                    conf_str = f"{d['confidence']:.1%}"
                    text_y   = row_y1 + int(row_h * 0.72)   # vertically inside row
                    cv2.putText(annotated, name_str, (tbl_x1 + pad, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, tbl_font_scale, tbl_text_color, tbl_thick)
                    cv2.putText(annotated, conf_str, (tbl_mid  + pad, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, tbl_font_scale, tbl_text_color, tbl_thick)

        
        # Draw overall status — top-right corner, compact
        status = "NOK" if result['is_defective'] else "OK"
        status_color = (0, 0, 255) if result['is_defective'] else (0, 255, 0)
        status_scale = 0.8 * scale
        status_thick = max(1, int(2 * scale))
        (sw, sh), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, status_scale, status_thick)
        sx = w - sw - int(10 * scale)
        sy = int(30 * scale)
        cv2.putText(annotated, status, (sx, sy),
                   cv2.FONT_HERSHEY_SIMPLEX, status_scale, status_color, status_thick)
        
        # Draw inference time (below status, right-aligned)
        time_str = f"{result['inference_time_ms']:.1f}ms"
        (tw, _), _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.putText(annotated, time_str, (w - tw - int(10 * scale), sy + sh + int(4 * scale)),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
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
            'NOK' if result['is_defective'] else 'OK',
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
            conf_threshold: Default per-zone confidence threshold (can be tuned per zone at runtime)
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
            status = 'NOK' if result['is_defective'] else 'OK'
            
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
        print("Press 'q' to quit | 's' to save snapshot | 'v' to toggle greyscale view | 'c' to capture to DB")
        print("="*50 + "\n")
        
        frame_count = 0
        fps_start = time.time()
        display_fps = 0.0
        show_grey = False          # V-key toggle: False = colour, True = greyscale
        
        window_name = "Defect Detection - Live Feed"
        settings_name = "Settings"
        show_settings = False
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while self._running:
                if show_settings:
                    try:
                        if cv2.getWindowProperty(settings_name, cv2.WND_PROP_VISIBLE) < 1:
                            show_settings = False
                        else:
                            for zi, zname in enumerate(self.detector.ZONE_NAMES):
                                pct = cv2.getTrackbarPos(f"Conf {zname} %", settings_name)
                                if pct is not None:
                                    self.detector.zone_conf[zi] = max(1, pct) / 100.0
                                    
                            area_val = cv2.getTrackbarPos("Min Area px", settings_name)
                            if area_val is not None:
                                self.detector.min_area_px = float(area_val)
                            
                            gpu_val = cv2.getTrackbarPos("GPU Preprocess", settings_name)
                            if gpu_val is not None:
                                self.detector.config.model.gpu_preprocessing = bool(gpu_val)
                    except cv2.error:
                        show_settings = False

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
                    from camera_capture import convert_to_greyscale, convert_to_greyscale_gpu
                    if self.detector.config.model.gpu_preprocessing:
                        grey_bgr = convert_to_greyscale_gpu(image, device=self.detector.device)
                    else:
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
                
                # Instruction to toggle settings
                cv2.putText(annotated, "[T] TO TOGGLE CONFIG SLIDERS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Show live feed
                cv2.imshow(window_name, annotated)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting live feed...")
                    break
                elif key == ord('v'):
                    show_grey = not show_grey
                    label = "GREYSCALE" if show_grey else "COLOUR"
                    print(f"  [V] Display mode switched to: {label}")
                elif key == ord('t') or key == 80:  # 'T' or Down Arrow
                    show_settings = not show_settings
                    if show_settings:
                        cv2.namedWindow(settings_name, cv2.WINDOW_NORMAL)
                        for zi, zname in enumerate(self.detector.ZONE_NAMES):
                            cv2.createTrackbar(f"Conf {zname} %", settings_name, int(self.detector.zone_conf[zi] * 100), 100, lambda x: None)
                        cv2.createTrackbar("Min Area px", settings_name, int(self.detector.min_area_px), 5000, lambda x: None)
                        cv2.createTrackbar("GPU Preprocess", settings_name, int(self.detector.config.model.gpu_preprocessing), 1, lambda x: None)
                    else:
                        try: cv2.destroyWindow(settings_name)
                        except Exception as e: print(f"Error: {e}")
                elif key == ord('s'):
                    # Save snapshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    status = 'NOK' if result['is_defective'] else 'OK'
                    save_dir = Path("logs/inspections") / status
                    save_dir.mkdir(parents=True, exist_ok=True)
                    snap_path = save_dir / f"snapshot_{status}_{timestamp}.png"
                    cv2.imwrite(str(snap_path), annotated)
                    print(f"  Snapshot saved: {snap_path}")
                
                elif key == ord('c'):
                    # Capture, save annotated image, and log it all to DB
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    status = 'NOK' if result['is_defective'] else 'OK'
                    save_dir = Path("logs/inspections") / status
                    save_dir.mkdir(parents=True, exist_ok=True)
                    snap_path = save_dir / f"snapshot_{status}_{timestamp}.png"
                    cv2.imwrite(str(snap_path), annotated)
                    
                    self.logger.log(result, str(snap_path))
                    print(f"  [C] Captured DB + Image: {status.upper()} ({result['defect_count']} defects). Saved to {snap_path}")
                
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
            image_raw = cv2.imread(args.source)
            annotated = detector.draw_results(image_raw, result)
            
            if result['is_defective']:
                print(f"REJECT - {result['defect_count']} defect(s)")
                for d in result['defects']:
                    print(f"  {d['class']}: {d['confidence']:.1%}")
            else:
                print("PASS - No defects")
            print(f"Inference time: {result['inference_time_ms']:.1f}ms")
            
            window_name = "Defect Detection - Image"
            settings_name = "Settings"
            show_settings = False
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            
            # Setup hover state
            state = {'hover_pos': None, 'last_hover_pos': (-1, -1)}
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_MOUSEMOVE:
                    state['hover_pos'] = (x, y)
                    
            cv2.setMouseCallback(window_name, mouse_callback)
            
            print("\nPress [T] to toggle config sliders | Any other key to close...")
            
            # Form initial display
            display_frame = detector.draw_results(image_raw, result, hover_pos=state['hover_pos'])
            cv2.putText(display_frame, "[T] TOGGLE CONFIG SLIDERS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow(window_name, display_frame)
            
            while True:
                pos = state['hover_pos']
                
                # Read per-zone sliders only when settings window is open
                zone_changed = False
                if show_settings:
                    try:
                        if cv2.getWindowProperty(settings_name, cv2.WND_PROP_VISIBLE) < 1:
                            show_settings = False
                        else:
                            for zi, zname in enumerate(detector.ZONE_NAMES):
                                pct = cv2.getTrackbarPos(f"Conf {zname} %", settings_name)
                                if pct is not None:
                                    new_conf = max(1, pct) / 100.0
                                    if new_conf != detector.zone_conf[zi]:
                                        detector.zone_conf[zi] = new_conf
                                        zone_changed = True
                            
                            area_val = cv2.getTrackbarPos("Min Area px", settings_name)
                            if area_val is not None:
                                new_area = float(area_val)
                                if new_area != detector.min_area_px:
                                    detector.min_area_px = new_area
                                    zone_changed = True
                            
                            gpu_val = cv2.getTrackbarPos("GPU Preprocess", settings_name)
                            if gpu_val is not None:
                                new_gpu = bool(gpu_val)
                                if new_gpu != detector.config.model.gpu_preprocessing:
                                    detector.config.model.gpu_preprocessing = new_gpu
                                    zone_changed = True
                    except cv2.error:
                        show_settings = False
                if zone_changed:
                    result = detector.detect(image_raw)
                    state['last_hover_pos'] = (-1, -1)  # force redraw
                        
                if pos != state['last_hover_pos']:
                    display_frame = detector.draw_results(image_raw, result, hover_pos=pos)
                    cv2.putText(display_frame, "[T] TOGGLE CONFIG SLIDERS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.imshow(window_name, display_frame)
                    state['last_hover_pos'] = pos
                    
                key = cv2.waitKey(30) & 0xFF
                if key == ord('t') or key == 80:  # T or Down Arrow
                    show_settings = not show_settings
                    if show_settings:
                        cv2.namedWindow(settings_name, cv2.WINDOW_NORMAL)
                        for zi, zname in enumerate(detector.ZONE_NAMES):
                            cv2.createTrackbar(f"Conf {zname} %", settings_name, int(detector.zone_conf[zi] * 100), 100, lambda x: None)
                        cv2.createTrackbar("Min Area px", settings_name, int(detector.min_area_px), 5000, lambda x: None)
                        cv2.createTrackbar("GPU Preprocess", settings_name, int(detector.config.model.gpu_preprocessing), 1, lambda x: None)
                    else:
                        try: cv2.destroyWindow(settings_name)
                        except Exception as e: print(f"Error: {e}")
                elif key != 255:
                    break
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
            
            window_name = "Defect Detection - Camera Capture"
            settings_name = "Settings"
            show_settings = False
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            
            # Setup hover state
            state = {'hover_pos': None, 'last_hover_pos': (-1, -1)}
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_MOUSEMOVE:
                    state['hover_pos'] = (x, y)
                    
            cv2.setMouseCallback(window_name, mouse_callback)
            
            print("\nPress [T] to toggle config sliders | 's' to save | Any other key to close...")
            
            display_frame = system.detector.draw_results(image, result, hover_pos=state['hover_pos'])
            cv2.putText(display_frame, "[T] TOGGLE CONFIG SLIDERS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow(window_name, display_frame)
            
            while True:
                pos = state['hover_pos']
                
                zone_changed = False
                if show_settings:
                    try:
                        if cv2.getWindowProperty(settings_name, cv2.WND_PROP_VISIBLE) < 1:
                            show_settings = False
                        else:
                            for zi, zname in enumerate(system.detector.ZONE_NAMES):
                                pct = cv2.getTrackbarPos(f"Conf {zname} %", settings_name)
                                if pct is not None:
                                    new_conf = max(1, pct) / 100.0
                                    if new_conf != system.detector.zone_conf[zi]:
                                        system.detector.zone_conf[zi] = new_conf
                                        zone_changed = True
                                        
                            area_val = cv2.getTrackbarPos("Min Area px", settings_name)
                            if area_val is not None:
                                new_area = float(area_val)
                                if new_area != system.detector.min_area_px:
                                    system.detector.min_area_px = new_area
                                    zone_changed = True
                                    
                            gpu_val = cv2.getTrackbarPos("GPU Preprocess", settings_name)
                            if gpu_val is not None:
                                new_gpu = bool(gpu_val)
                                if new_gpu != system.detector.config.model.gpu_preprocessing:
                                    system.detector.config.model.gpu_preprocessing = new_gpu
                                    zone_changed = True
                    except cv2.error:
                        show_settings = False
                if zone_changed:
                    result = system.detector.detect(image)
                    state['last_hover_pos'] = (-1, -1)  # force redraw
                        
                if pos != state['last_hover_pos']:
                    display_frame = system.detector.draw_results(image, result, hover_pos=pos)
                    cv2.putText(display_frame, "[T] TOGGLE CONFIG SLIDERS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.imshow(window_name, display_frame)
                    state['last_hover_pos'] = pos
                    
                key = cv2.waitKey(30) & 0xFF
                if key == ord('t') or key == 80:  # T or Down Arrow
                    show_settings = not show_settings
                    if show_settings:
                        cv2.namedWindow(settings_name, cv2.WINDOW_NORMAL)
                        for zi, zname in enumerate(system.detector.ZONE_NAMES):
                            cv2.createTrackbar(f"Conf {zname} %", settings_name, int(system.detector.zone_conf[zi] * 100), 100, lambda x: None)
                        cv2.createTrackbar("Min Area px", settings_name, int(system.detector.min_area_px), 5000, lambda x: None)
                        cv2.createTrackbar("GPU Preprocess", settings_name, int(system.detector.config.model.gpu_preprocessing), 1, lambda x: None)
                    else:
                        try: cv2.destroyWindow(settings_name)
                        except Exception as e: print(f"Error: {e}")
                elif key != 255:
                    if key == ord('s'):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = f"logs/capture_{timestamp}.png"
                        os.makedirs("logs", exist_ok=True)
                        cv2.imwrite(save_path, system.detector.draw_results(image, result))
                        print(f"Saved: {save_path}")
                    break
                    
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

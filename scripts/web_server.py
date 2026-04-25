r"""
=============================================================
  web_server.py  --  FastAPI web interface for DefectDetector
=============================================================
Replaces cv2.imshow with a browser-based live feed.

HOW TO USE
----------
1. Install dependencies (once):
       pip install fastapi uvicorn[standard]

2. Run the server:
       python web_server.py --model runs/segment/models/my_seg_run7/weights/best.engine

   Optional flags:
       --port   8000          (default)
       --conf   0.03          per-zone confidence threshold
       --mock                 use mock camera instead of Basler
       --source path/to.jpg   single image mode (for testing without camera)

3. Open your browser on any device on the same network:
       http://localhost:8000          (on the same PC)
       http://192.168.x.x:8000       (from any factory device)

ARCHITECTURE
------------
  Main thread   : FastAPI / uvicorn HTTP server
  Worker thread : Camera capture -> YOLO inference -> frame encoding
  Shared state  : FrameBuffer (thread-safe) + InspectionState
=============================================================
"""

import argparse
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# Import your existing detector
from defect_detector import DefectDetector, InspectionLogger
from camera_capture import get_camera, convert_to_greyscale


# =============================================================================
#  Shared state (accessed by inference thread + HTTP handlers)
# =============================================================================

class FrameBuffer:
    """Thread-safe single-frame buffer.
    Inference thread writes; MJPEG generator reads the latest JPEG."""
    def __init__(self):
        self._lock  = threading.Lock()
        self._frame: Optional[bytes] = None

    def push(self, jpeg_bytes: bytes):
        with self._lock:
            self._frame = jpeg_bytes

    def get(self) -> Optional[bytes]:
        with self._lock:
            return self._frame


class InspectionState:
    """Mutable shared state: latest result + rolling inspection history."""
    def __init__(self, history_len: int = 200):
        self._lock         = threading.Lock()
        self.latest_result = {}
        self.history       = deque(maxlen=history_len)
        self.fps           = 0.0
        self.is_running    = False
        self.show_grey     = False

    def push_result(self, result: dict, fps: float):
        entry = {
            "ts":           result.get("timestamp", datetime.now().isoformat()),
            "is_defective": result.get("is_defective", False),
            "defect_count": result.get("defect_count", 0),
            "inference_ms": result.get("inference_time_ms", 0),
        }
        with self._lock:
            self.latest_result = result
            self.history.append(entry)
            self.fps = fps

    def snapshot(self) -> dict:
        with self._lock:
            h      = list(self.history)
            total  = len(h)
            nok    = sum(1 for e in h if e["is_defective"])
            ok     = total - nok
            avg_ms = (sum(e["inference_ms"] for e in h) / total) if total else 0
            return {
                "total":      total,
                "nok":        nok,
                "ok":         ok,
                "nok_rate":   round(nok / total * 100, 1) if total else 0,
                "avg_ms":     round(avg_ms, 1),
                "fps":        round(self.fps, 1),
                "history":    h[-50:],
                "latest":     self.latest_result,
                "is_running": self.is_running,
            }


FRAME_BUF = FrameBuffer()
STATE     = InspectionState()
DETECTOR: Optional[DefectDetector] = None
LOGGER    = InspectionLogger()


# =============================================================================
#  Inference worker thread
# =============================================================================

def _inference_loop(model_path: str, config_path: str,
                    conf: float, mock: bool, source_image: Optional[str]):
    """Runs in a daemon thread. Continuously captures -> detects -> encodes."""
    global DETECTOR, STATE

    DETECTOR = DefectDetector(model_path, config_path, conf_threshold=conf)

    # -- Single-image mode (no camera, useful for testing) -------------------
    if source_image:
        img = cv2.imread(source_image)
        if img is None:
            print(f"[web_server] ERROR: Cannot read image: {source_image}")
            return
        result    = DETECTOR.detect(img)
        annotated = DETECTOR.draw_results(img, result, always_show_labels=False)
        _, jpeg   = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        FRAME_BUF.push(jpeg.tobytes())
        STATE.push_result(result, fps=0.0)
        STATE.is_running = True
        print("[web_server] Static image mode. Open http://localhost:8000")
        while True:          # keep thread alive so server can serve the frame
            time.sleep(60)

    # -- Live camera mode ----------------------------------------------------
    camera = get_camera(config_path, use_mock=mock)
    if not camera.connect():
        print("[web_server] ERROR: Camera connection failed.")
        return

    camera.start_streaming()
    STATE.is_running = True
    print("[web_server] Camera streaming started. Open http://localhost:8000")

    frame_count = 0
    fps_clock   = time.time()
    display_fps = 0.0

    try:
        while True:
            frame = camera.capture()
            if frame is None:
                time.sleep(0.1)
                continue

            # Downsample massive camera frames proportionally to speed up CPU ops
            # (CLAHE, contour finding, JPEG encoding) while keeping aspect ratio
            h, w = frame.shape[:2]
            target_w = 1024
            if w > target_w:
                target_h = int(h * (target_w / w))
                frame = cv2.resize(frame, (target_w, target_h))

            display_frame = convert_to_greyscale(frame) if STATE.show_grey else frame
            result        = DETECTOR.detect(frame)
            annotated     = DETECTOR.draw_results(display_frame, result, always_show_labels=False)

            # FPS overlay
            frame_count += 1
            elapsed = time.time() - fps_clock
            if elapsed >= 1.0:
                display_fps = frame_count / elapsed
                frame_count = 0
                fps_clock   = time.time()

            cv2.putText(annotated, f"FPS: {display_fps:.1f}",
                        (10, annotated.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
            FRAME_BUF.push(jpeg.tobytes())
            STATE.push_result(result, display_fps)

    except Exception as exc:
        print(f"[web_server] Inference thread error: {exc}")
    finally:
        camera.disconnect()
        STATE.is_running = False


# =============================================================================
#  FastAPI application
# =============================================================================

app = FastAPI(title="Defect Detection System", version="1.0")


# -- MJPEG stream -------------------------------------------------------------

def _mjpeg_generator():
    """Yields multipart JPEG frames consumed by the browser <img> tag."""
    boundary = b"--frame\r\n"
    while True:
        jpeg = FRAME_BUF.get()
        if jpeg is None:
            time.sleep(0.05)
            continue
        yield (
            boundary
            + b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg
            + b"\r\n"
        )
        time.sleep(0.033)   # ~30 fps cap for the HTTP stream


@app.get("/feed")
def video_feed():
    """MJPEG live stream endpoint."""
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# -- Stats --------------------------------------------------------------------

@app.get("/api/stats")
def get_stats():
    return JSONResponse(STATE.snapshot())


# -- Get current thresholds ---------------------------------------------------

@app.get("/api/thresholds")
def get_thresholds():
    if DETECTOR is None:
        return JSONResponse({"error": "Detector not ready"}, status_code=503)
    return JSONResponse({
        "zone_conf":     [round(c, 2) for c in DETECTOR.zone_conf],
        "zone_names":    DETECTOR.ZONE_NAMES,
        "min_area_px":   DETECTOR.min_area_px,
        "iou_threshold": DETECTOR.iou_threshold,
    })


# -- Update thresholds --------------------------------------------------------

@app.post("/api/thresholds")
async def set_thresholds(request: Request):
    """
    Body JSON:
      { "zone_conf": [0.3, 0.3, 0.5, 0.25],
        "min_area_px": 150,
        "iou_threshold": 0.45 }
    """
    body = await request.json()
    if DETECTOR is None:
        return JSONResponse({"error": "Detector not ready"}, status_code=503)
    if "zone_conf" in body:
        vals = body["zone_conf"]
        if isinstance(vals, list) and len(vals) == 4:
            DETECTOR.zone_conf = [max(0.01, min(1.0, float(v))) for v in vals]
    if "min_area_px" in body:
        DETECTOR.min_area_px = float(body["min_area_px"])
    if "iou_threshold" in body:
        DETECTOR.iou_threshold = float(body["iou_threshold"])
    return JSONResponse({"status": "ok"})


# -- Toggle greyscale display -------------------------------------------------

@app.post("/api/toggle_grey")
def toggle_grey():
    STATE.show_grey = not STATE.show_grey
    return JSONResponse({"show_grey": STATE.show_grey})


# -- Snapshot -----------------------------------------------------------------

@app.post("/api/snapshot")
def snapshot():
    jpeg = FRAME_BUF.get()
    if jpeg is None:
        return JSONResponse({"error": "No frame available"}, status_code=503)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_dir = Path("logs/snapshots")
    save_dir.mkdir(parents=True, exist_ok=True)
    path     = save_dir / f"snap_{ts}.jpg"
    path.write_bytes(jpeg)
    
    # Log the snapshot and the current detection result to the SQLite DB
    # so it automatically appears in the Streamlit dashboard
    if STATE.latest_result:
        res_copy = STATE.latest_result.copy()
        res_copy['timestamp'] = datetime.now().isoformat()
        LOGGER.log(res_copy, image_path=str(path))
        
    return JSONResponse({"saved": str(path)})


# -- Serve single-page HTML ---------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# =============================================================================
#  Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Defect Detection Web Server")
    parser.add_argument("--model",  required=True,
                        help="Path to .engine / .pt model weights")
    parser.add_argument("--config", default="config/system_config.json")
    parser.add_argument("--conf",   type=float, default=0.03,
                        help="Per-zone confidence threshold (default: 0.03)")
    parser.add_argument("--port",   type=int,   default=8000)
    parser.add_argument("--host",   default="0.0.0.0",
                        help="Bind host (0.0.0.0 = accessible on LAN)")
    parser.add_argument("--mock",   action="store_true",
                        help="Use mock camera (no hardware needed)")
    parser.add_argument("--source", default=None,
                        help="Path to a single image file (skips camera, for testing)")
    args = parser.parse_args()

    # Start inference in a background daemon thread
    t = threading.Thread(
        target=_inference_loop,
        args=(args.model, args.config, args.conf, args.mock, args.source),
        daemon=True,
        name="InferenceWorker",
    )
    t.start()

    print(f"\n{'='*55}")
    print(f"  Defect Detection Web Server")
    print(f"  Local :  http://localhost:{args.port}")
    print(f"  LAN   :  http://<this-pc-ip>:{args.port}")
    print(f"{'='*55}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

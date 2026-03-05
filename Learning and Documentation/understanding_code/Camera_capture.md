# 📷 Camera Capture — Complete Code Walkthrough
**In an industrial project like this, 

camera_capture.py
 is the "Eyes" of the system. Without it, you have no data to train on and no live images to inspect.

Here are the 4 main reasons why we need this specific file:

1. Hardware Communication (The "Bridge")
Basler cameras are industrial-grade sensors. They don't appear as a "standard camera" in Windows like a Logitech webcam does. You can't just open them in the Windows Camera app.

The problem: The camera speaks a complex hardware language (GigE Vision or USB3 Vision).
The solution: This 

.py
 file uses the pypylon SDK to bridge the gap between that complex hardware and Python. It acts as the "translator."
2. Precise Control (Consistency)
For AI to work well, the images must be consistent. If one image is bright and the next is dark, the YOLO model will get confused.

Manual capture: A phone camera "thinks" for you—it auto-adjusts brightness and focus, which changes the data.
This script: It locks the Exposure (brightness) and Gain (noise) to exact values. Every single photo will have the exact same lighting conditions, which makes the AI much more accurate.
3. Automated Labeling (Speed)
When you are building a dataset, you need hundreds of images.

Manual way: Take a photo → rename the file to "good_1.png" → move it to a folder. (Very slow!)
This script: In Interactive Mode, you just press 'g'. The script takes the photo, generates a timestamp, and saves it directly into the dataset/raw/good/ folder for you. You can collect a 500-image dataset in minutes.
4. The "Mock" Fallback (Development)
This is a unique feature of our script.

The problem: If you go home and don't have the $500 Basler camera with you, your code would normally crash.
The solution: The MockCamera class in this file detects that no camera is connected and "fakes" a camera. This allows you to keep coding and testing your training pipeline anywhere, even without the hardware.
**

> **Script**: `scripts/camera_capture.py`  
> **Purpose**: Interface with Basler industrial cameras to capture product images for the defect detection dataset.  
> **When to use**: During the data collection phase — Step 1 of the pipeline.

---

## Table of Contents

- [Overview](#overview)
- [Imports & Dependencies](#imports--dependencies)
- [BaslerCamera Class](#baslercamera-class)
  - [Initialization](#initialization--__init__)
  - [Loading Config](#loading-config--_load_config)
  - [Connecting to Camera](#connecting-to-camera--connect)
  - [Applying Settings](#applying-settings--_apply_settings)
  - [Capturing an Image](#capturing-an-image--capture)
  - [Capture & Save](#capture--save--capture_and_save)
  - [Disconnecting](#disconnecting--disconnect)
  - [Context Manager](#context-manager--__enter__--__exit__)
- [MockCamera Class](#mockcamera-class)
- [Factory Function — get_camera()](#factory-function--get_camera)
- [Interactive Collection Mode](#interactive-collection-mode)
- [How to Run](#how-to-run)
- [How It Connects to Other Scripts](#how-it-connects-to-other-scripts)

---

## Overview

This script has **3 main components**:

```
┌───────────────────────────────────────────────────────┐
│                   camera_capture.py                    │
│                                                        │
│  ┌─────────────┐   ┌─────────────┐   ┌────────────┐  │
│  │BaslerCamera  │   │ MockCamera  │   │ Interactive│  │
│  │(real camera) │   │ (testing)   │   │ Collection │  │
│  └──────┬──────┘   └──────┬──────┘   │   Mode     │  │
│         │                  │          └─────┬──────┘  │
│         └──────┬───────────┘                │         │
│                ▼                             │         │
│         get_camera()  ◄─────────────────────┘         │
│         (factory fn)                                   │
└───────────────────────────────────────────────────────┘
```

1. **`BaslerCamera`** — Talks to real Basler hardware via the pypylon SDK
2. **`MockCamera`** — Fake camera for testing without hardware
3. **`collect_dataset_interactive()`** — Interactive mode where you press keys to capture & label images

---

## Imports & Dependencies

```python
"""                                                  # Lines 1-4
Basler Camera Capture Utility                        # Docstring — describes the script's purpose
Captures images from Basler camera for defect        # Python convention: every file should have
detection dataset collection.                        # a module-level docstring
"""

import os                                            # Line 6: File system operations (checking paths)
import json                                          # Line 7: Parse JSON config files
from datetime import datetime                        # Line 8: Timestamps for unique filenames
from pathlib import Path                             # Line 9: Modern way to handle file paths
```

### The pypylon Import (Lines 11–16) — Graceful Degradation Pattern

```python
try:                                                 # Line 11
    from pypylon import pylon                        # Line 12: Try to import Basler SDK
    PYPYLON_AVAILABLE = True                         # Line 13: Flag: SDK is installed ✓
except ImportError:                                  # Line 14: If pypylon isn't installed...
    PYPYLON_AVAILABLE = False                        # Line 15: Flag: SDK is missing ✗
    print("Warning: pypylon not installed...")       # Line 16: Warn the user (but don't crash)
```

> **💡 Why this pattern?**
> 
> Without `try/except`, if pypylon isn't installed, the **entire script would crash** on import — even if you only wanted to use `MockCamera`. This pattern lets the script load successfully either way. The flag `PYPYLON_AVAILABLE` is checked later to decide which camera class to use.
> 
> **This is a common pattern in Python for optional dependencies.**

```python
import cv2                                           # Line 18: OpenCV — image reading, writing, display
import numpy as np                                   # Line 19: NumPy — images are stored as NumPy arrays
```

> **💡 What is a NumPy array?**
> 
> Every image in OpenCV is a NumPy array with shape `(height, width, channels)`.  
> For a 640×480 color image: shape = `(480, 640, 3)` — 3 channels = Blue, Green, Red (BGR).  
> Each pixel value is 0–255 (8-bit unsigned integer).

---

## BaslerCamera Class

### Initialization — `__init__`

```python
class BaslerCamera:                                  # Line 22
    """Wrapper class for Basler camera operations"""  # Line 23: Class docstring

    def __init__(self, config_path: str = None):     # Line 25: Constructor
        """                                          # Lines 26-31: Docstring
        Initialize Basler camera

        Args:
            config_path: Path to system_config.json (optional)
        """
        if not PYPYLON_AVAILABLE:                    # Line 32: Safety check
            raise ImportError(                       # Line 33: If SDK isn't installed,
                "pypylon is not installed..."        #           raise an error immediately
            )

        # Load configuration                        # Line 35: Comment
        self.config = self._load_config(config_path) # Line 36: Load camera settings from JSON
        self.camera = None                           # Line 37: Camera object — None until connected
        self.converter = None                        # Line 38: Image format converter — set on connect
```

> **💡 Key concepts:**
>
> - `self` refers to the current instance of the class. Every method gets `self` as its first argument.
> - `self.camera = None` means the camera isn't connected yet. We set it to `None` as a sentinel value — you can check `if self.camera is None:` later.
> - `config_path: str = None` means this parameter is **optional**. If not provided, it defaults to `None`.

---

### Loading Config — `_load_config`

```python
    def _load_config(self, config_path: str) -> dict:    # Line 40
        """Load camera configuration from JSON file"""    # Line 41

        # Default values if no config file is provided
        default_config = {                                # Line 42
            "exposure_time_us": 15000,                    # Line 43: 15,000 microseconds = 15ms
            "gain_db": 0,                                 # Line 44: No amplification
            "pixel_format": "Mono8",                      # Line 45: Grayscale, 8 bits per pixel
            "trigger_mode": "Software",                   # Line 46: Capture triggered by code
            "timeout_ms": 5000                            # Line 47: Wait max 5 seconds
        }

        if config_path and os.path.exists(config_path):  # Line 50: If config file exists...
            with open(config_path, 'r') as f:            # Line 51: Open and read it
                full_config = json.load(f)                # Line 52: Parse JSON → Python dict
                return full_config.get('camera',          # Line 53: Return the "camera" section
                                       default_config)   #           or defaults if missing

        return default_config                             # Line 55: No config file → use defaults
```

> **💡 What each camera setting means:**
>
> | Setting | Value | What It Controls |
> |---------|-------|------------------|
> | `exposure_time_us` | 15000 | How long the sensor collects light (15ms). Higher = brighter but more motion blur |
> | `gain_db` | 0 | Electronic signal amplification. Higher = brighter but more noise. 0 = cleanest image |
> | `pixel_format` | Mono8 | Grayscale, 8-bit (256 shades of gray). Sufficient for defect detection |
> | `trigger_mode` | Software | The code tells the camera when to capture (vs. external trigger signal) |
> | `timeout_ms` | 5000 | If camera doesn't respond in 5 seconds, throw an error |
>
> **Why the underscore in `_load_config`?**  
> The leading `_` is a Python convention meaning "private method" — it's used internally by the class and shouldn't be called directly from outside. It signals to other developers: "This is an implementation detail, not part of the public API."

---

### Connecting to Camera — `connect`

```python
    def connect(self) -> bool:                           # Line 57: Returns True/False
        """Connect to the first available Basler camera"""

        try:                                             # Line 59
            # Get the first available camera
            self.camera = pylon.InstantCamera(           # Line 61: Create camera object
                pylon.TlFactory.GetInstance()             # Line 62: TlFactory = Transport Layer Factory
                    .CreateFirstDevice()                  #           Finds the first Basler camera
            )                                            #           connected via USB or GigE
            self.camera.Open()                           # Line 64: Open connection to camera hardware
```

> **💡 What's happening under the hood:**
>
> ```
> Your Computer                         Basler Camera
> ┌──────────────┐                     ┌────────────┐
> │ Python code  │                     │            │
> │      │       │  USB / GigE cable   │   Sensor   │
> │ TlFactory ───│─────────────────────│── Lens     │
> │      │       │                     │            │
> │ InstantCamera│                     └────────────┘
> │      │       │
> │    Open() ───│── "Hello camera, I want to talk"
> └──────────────┘
> ```
>
> `TlFactory` (Transport Layer Factory) handles the communication protocol. `CreateFirstDevice()` scans all connections and grabs the first Basler camera it finds.

```python
            # Apply settings
            self._apply_settings()                       # Line 67: Set exposure, gain, etc.

            # Create image converter for consistent output
            self.converter = pylon.ImageFormatConverter() # Line 70: Create converter
            self.converter.OutputPixelFormat = \
                pylon.PixelType_BGR8packed                # Line 71: Output in BGR format
            self.converter.OutputBitAlignment = \
                pylon.OutputBitAlignment_MsbAligned       # Line 72: Standard bit alignment

            print(f"Connected to camera: "
                  f"{self.camera.GetDeviceInfo()"
                  f".GetModelName()}")                   # Line 74: Print camera model name
            return True                                  # Line 75: Success!

        except Exception as e:                           # Line 77
            print(f"Failed to connect to camera: {e}")   # Line 78: Connection failed
            return False                                 # Line 79: Return False (caller can handle)
```

> **💡 Why convert to BGR?**
>
> Basler cameras output in various raw formats (Mono8, BayerRG, YUV, etc.). OpenCV expects **BGR** (Blue-Green-Red) format. The `ImageFormatConverter` translates between them so all downstream code can use standard OpenCV operations like `cv2.imshow()` and `cv2.imwrite()`.

---

### Applying Settings — `_apply_settings`

```python
    def _apply_settings(self):                           # Line 81
        """Apply configuration settings to camera"""
        try:                                             # Line 83
            # Exposure time (microseconds)
            if self.camera.ExposureTime.IsWritable():    # Line 85: Check if setting is changeable
                self.camera.ExposureTime.SetValue(       # Line 86: Set exposure time
                    self.config['exposure_time_us']
                )

            # Gain (dB)
            if self.camera.Gain.IsWritable():            # Line 89: Check if gain is changeable
                self.camera.Gain.SetValue(               # Line 90: Set gain value
                    self.config['gain_db']
                )

            print(f"Camera settings applied:")           # Line 92-94: Log what was set
            print(f"  - Exposure: {self.config['exposure_time_us']} µs")
            print(f"  - Gain: {self.config['gain_db']} dB")

        except Exception as e:                           # Line 96-97
            print(f"Warning: Could not apply all settings: {e}")
```

> **💡 Why `IsWritable()` check?**
>
> Some cameras are locked (e.g., in continuous capture mode) or don't support certain features. Trying to write to a read-only property would crash the program. `IsWritable()` prevents this — it's a **defensive programming** technique.

---

### Capturing an Image — `capture`

```python
    def capture(self) -> np.ndarray:                     # Line 99: Returns a NumPy array (image)
        """Capture a single image from the camera"""

        if self.camera is None:                          # Line 106: Safety check
            print("Error: Camera not connected")
            return None                                  # Return None (caller must handle)

        try:                                             # Line 110
            self.camera.StartGrabbing(                   # Line 111: Begin capturing mode
                pylon.GrabStrategy_LatestImageOnly       #   Strategy: only keep the newest frame
            )                                            #   (discard old frames if we're slow)

            grab_result = self.camera.RetrieveResult(    # Line 113: Wait for a frame
                self.config['timeout_ms'],               # Line 114: Timeout (5000ms = 5 seconds)
                pylon.TimeoutHandling_ThrowException      # Line 115: If timeout → throw exception
            )

            if grab_result.GrabSucceeded():              # Line 118: Did we get a valid frame?
                # Convert to OpenCV format
                image = self.converter.Convert(           # Line 120: Raw camera data → BGR
                    grab_result
                )
                img_array = image.GetArray()              # Line 121: → NumPy array

                grab_result.Release()                    # Line 123: Free the camera buffer
                self.camera.StopGrabbing()               # Line 124: Stop capture mode

                return img_array                         # Line 126: Return the image!

            else:                                        # Line 127: Grab failed
                print(f"Grab failed: "
                      f"{grab_result.ErrorCode}")        # Line 128: Print error code
                grab_result.Release()                    # Line 129: Clean up
                self.camera.StopGrabbing()               # Line 130
                return None                              # Line 131

        except Exception as e:                           # Line 133
            print(f"Capture error: {e}")                 # Line 134
            return None                                  # Line 135
```

> **💡 Understanding the capture flow:**
>
> ```
> StartGrabbing()          "Camera, start capturing continuously"
>       │
>       ▼
> RetrieveResult()         "Give me the latest frame (wait up to 5s)"
>       │
>       ▼
> GrabSucceeded()?         "Was the frame captured correctly?"
>       │
>   Yes ▼           No ▼
> Convert()           Return None
>       │
>       ▼
> GetArray()           "Convert the Basler format → NumPy array"
>       │
>       ▼
> Release()            "Free the memory buffer"
> StopGrabbing()       "Camera, stop capturing"
>       │
>       ▼
> Return image         "Here's your image as a (H, W, 3) NumPy array"
> ```
>
> **Why `GrabStrategy_LatestImageOnly`?**  
> Without this, the camera buffers frames in a queue. If your code is slow, you'd process old frames. `LatestImageOnly` always gives you the newest frame — critical for real-time inspection where you want the current product, not one from 2 seconds ago.

---

### Capture & Save — `capture_and_save`

```python
    def capture_and_save(self, output_dir: str,          # Line 137
                         label: str,                      # Line 138
                         metadata: dict = None) -> str:   #   Returns filepath

        image = self.capture()                           # Line 150: Take a photo

        if image is None:                                # Line 152: Did it work?
            return None                                  #   No → return None

        # Create output directory
        save_dir = Path(output_dir) / label              # Line 156: e.g., "dataset/raw/good"
        save_dir.mkdir(parents=True, exist_ok=True)      # Line 157: Create folder if needed
        #         parents=True  → create parent dirs too
        #         exist_ok=True → don't error if it exists

        # Generate filename with timestamp
        timestamp = datetime.now().strftime(             # Line 160
            "%Y%m%d_%H%M%S_%f"                           #   Format: 20260216_143021_123456
        )                                                #   %f = microseconds (for uniqueness)
        filename = f"{label}_{timestamp}.png"            # Line 161: e.g., "good_20260216_143021_123456.png"
        filepath = save_dir / filename                   # Line 162: Full path

        # Save image (PNG for lossless quality)
        cv2.imwrite(str(filepath), image)                # Line 165: Write image to disk

        print(f"Saved: {filepath}")                      # Line 167
        return str(filepath)                             # Line 168: Return the path
```

> **💡 Why PNG and not JPEG?**
>
> | Format | Compression | Quality Loss | Use Case |
> |--------|-------------|-------------|----------|
> | **PNG** | Lossless | None — pixel-perfect | Training data (preserves defect details) |
> | **JPEG** | Lossy | Yes — compression artifacts | Web/preview (smaller file size) |
>
> For training data, PNG is essential because JPEG artifacts could be mistaken for defects, or could hide real defects. A blurry compression artifact near a scratch could confuse the model.

---

### Disconnecting — `disconnect`

```python
    def disconnect(self):                                # Line 170
        """Disconnect from camera"""
        if self.camera is not None:                      # Line 172: Only if connected
            self.camera.Close()                          # Line 173: Close hardware connection
            self.camera = None                           # Line 174: Reset to None
            print("Camera disconnected")                 # Line 175
```

> **💡 Why set `self.camera = None`?**
>
> This prevents accidentally trying to use a closed camera. Any subsequent calls to `capture()` will hit the `if self.camera is None:` safety check and return `None` gracefully instead of crashing.

---

### Context Manager — `__enter__` / `__exit__`

```python
    def __enter__(self):                                 # Line 177: Called when entering `with` block
        """Context manager entry"""
        self.connect()                                   # Line 179: Auto-connect
        return self                                      # Line 180: Return self for use in `with` body

    def __exit__(self, exc_type, exc_val, exc_tb):      # Line 182: Called when leaving `with` block
        """Context manager exit"""
        self.disconnect()                                # Line 184: Auto-disconnect (even on errors!)
```

> **💡 What is a Context Manager?**
>
> It lets you write this:
> ```python
> # WITHOUT context manager (risky — might forget to disconnect)
> camera = BaslerCamera()
> camera.connect()
> image = camera.capture()
> camera.disconnect()         # What if capture() crashes? This line is skipped!
>
> # WITH context manager (safe — always disconnects)
> with BaslerCamera() as camera:
>     image = camera.capture()
> # disconnect() is GUARANTEED to run, even if an error occurs
> ```
>
> The `exc_type, exc_val, exc_tb` parameters tell you about any exception that occurred inside the `with` block. We don't use them here — we just disconnect regardless.
>
> **This is a Python best practice for any resource that needs cleanup** (files, database connections, cameras, network sockets).

---

## MockCamera Class

**Purpose**: A fake camera that generates random images. Used for development and testing when no physical Basler camera is available.

```python
class MockCamera:                                        # Line 187
    """Mock camera for testing without hardware"""

    def __init__(self, config_path: str = None):         # Line 190
        self.config = {"exposure_time_us": 15000,        # Line 191: Minimal config
                       "gain_db": 0}
        print("Using MockCamera (no hardware connected)")# Line 192

    def connect(self) -> bool:                           # Line 194
        print("MockCamera connected (simulated)")        # Line 195: Just prints a message
        return True                                      # Line 196: Always "succeeds"

    def capture(self) -> np.ndarray:                     # Line 198
        """Generate a test pattern image"""
        # Create a 640x480 test image with random noise
        image = np.random.randint(                       # Line 201
            100, 200,                                    #   Pixel values between 100-200 (gray-ish)
            (480, 640, 3),                               #   Shape: height=480, width=640, channels=3
            dtype=np.uint8                               #   Data type: unsigned 8-bit integer (0-255)
        )

        # Add some simulated "defects" (random dark spots)
        if np.random.random() > 0.5:                     # Line 204: 50% chance of adding a defect
            x = np.random.randint(50, 590)               # Line 205: Random X position
            y = np.random.randint(50, 430)               #           Random Y position
            cv2.rectangle(image,                         # Line 206: Draw a dark rectangle
                          (x, y),                        #           Top-left corner
                          (x+50, y+20),                  #           Bottom-right corner (50×20 pixels)
                          (50, 50, 50),                  #           Color: dark gray
                          -1)                            #           -1 = filled (not just outline)

        return image                                     # Line 208: Return the fake image
```

> **💡 Why does MockCamera exist?**
>
> - You can develop the full pipeline on a laptop without a $500+ Basler camera
> - Unit tests can run without hardware
> - CI/CD pipelines (GitHub Actions, etc.) can test the code automatically
> - New team members can experiment without access to the camera
>
> The mock generates a gray noisy image with occasional dark rectangles as "defects" — just enough to verify the downstream pipeline works.

---

## Factory Function — `get_camera()`

```python
def get_camera(config_path: str = None,                  # Line 236
               use_mock: bool = False):
    """
    Factory function to get appropriate camera instance

    Args:
        config_path: Path to configuration file
        use_mock: Force use of mock camera

    Returns:
        Camera instance (BaslerCamera or MockCamera)
    """
    if use_mock or not PYPYLON_AVAILABLE:                 # Line 247
        return MockCamera(config_path)                   # Line 248: No SDK → use mock
    return BaslerCamera(config_path)                     # Line 249: SDK available → use real
```

> **💡 What is a Factory Function?**
>
> A **factory** creates objects without exposing the creation logic. Other code just calls `get_camera()` and gets back the right camera type — it doesn't need to know about pypylon, SDK availability, or hardware.
>
> ```python
> # Other scripts use this pattern:
> camera = get_camera()    # Automatically picks real or mock
>
> # Instead of:
> if PYPYLON_AVAILABLE:
>     camera = BaslerCamera()
> else:
>     camera = MockCamera()
> # ↑ This logic is hidden inside the factory
> ```
>
> This is the **Factory Pattern** — one of the most common design patterns in software engineering.

---

## Interactive Collection Mode

```python
def collect_dataset_interactive():                       # Line 253
    """Interactive mode for collecting dataset images"""

    print("=" * 50)                                      # Line 258: Print header
    print("DATASET COLLECTION MODE")
    print("=" * 50)
    print("Press keys to capture images:")
    print("  g - Good product")                          # Line 262: Key mappings
    print("  s - Scratch defect")
    print("  c - Crack defect")
    print("  d - Dent defect")
    print("  x - Discoloration/Other defect")
    print("  q - Quit")
    print("=" * 50)

    output_dir = "dataset/raw"                           # Line 270: Where to save images

    camera = get_camera(                                 # Line 272: Auto-select camera type
        use_mock=not PYPYLON_AVAILABLE                   #   Mock if no SDK, real if available
    )

    with camera:                                         # Line 274: Context manager → auto connect/disconnect
        while True:                                      # Line 275: Infinite loop until 'q'
            key = input(                                 # Line 276: Wait for user input
                "Enter label key (g/s/c/d/x) or q: "
            ).lower().strip()                            #   .lower() → case-insensitive
                                                         #   .strip() → remove whitespace

            label_map = {                                # Line 278: Map keys to label names
                'g': 'good',
                's': 'scratch',
                'c': 'crack',
                'd': 'dent',
                'x': 'other'
            }

            if key == 'q':                               # Line 286: Quit
                print("Exiting collection mode...")
                break                                    # Line 288: Exit the while loop

            elif key in label_map:                       # Line 289: Valid key pressed
                label = label_map[key]                   # Line 290: Get label name
                filepath = camera.capture_and_save(      # Line 291: Capture + save
                    output_dir, label
                )
                if filepath:                             # Line 292: Success?
                    print(f"✓ Captured {label} image")   # Line 293: Confirm

            else:                                        # Line 294: Invalid key
                print("Invalid key. Use g/s/c/d/x "
                      "or q to quit.")                   # Line 295
```

> **💡 The collection workflow:**
>
> ```
>  ┌────────────┐      ┌───────────┐      ┌──────────────┐
>  │ Place      │      │ Press key │      │ Image saved  │
>  │ product on │─────▶│ g/s/c/d/x │─────▶│ dataset/raw/ │
>  │ camera     │      │           │      │ {label}/     │
>  └────────────┘      └───────────┘      └──────────────┘
>        │                                       │
>        └───────────── Repeat ──────────────────┘
> ```
>
> Files are saved as:
> ```
> dataset/raw/
> ├── good/
> │   ├── good_20260216_143021_123456.png
> │   └── good_20260216_143029_654321.png
> ├── scratch/
> │   └── scratch_20260216_143045_789012.png
> └── crack/
>     └── crack_20260216_143102_345678.png
> ```

---

### Script Entry Point

```python
if __name__ == "__main__":                               # Line 298
    collect_dataset_interactive()                        # Line 300: Run interactive mode
```

> **💡 What does `if __name__ == "__main__":` mean?**
>
> - When you run `python camera_capture.py` directly → `__name__` is `"__main__"` → this block runs
> - When you do `from camera_capture import BaslerCamera` (import from another script) → `__name__` is `"camera_capture"` → this block is **skipped**
>
> This lets the file work as both a standalone script AND an importable module. Other scripts like `defect_detector.py` import `BaslerCamera` and `MockCamera` from this file without triggering the interactive mode.

---

## How to Run

```bash
# Interactive collection mode (default)
python scripts/camera_capture.py

# With real camera (requires Basler Pylon SDK + pypylon)
# 1. Install Pylon SDK from https://www.baslerweb.com/en/downloads/
# 2. pip install pypylon
# 3. Connect Basler camera via USB/GigE
# 4. python scripts/camera_capture.py

# Without camera (auto-uses MockCamera)
# Just run it — if pypylon isn't installed, MockCamera is used automatically
```

---

## How It Connects to Other Scripts

```
camera_capture.py
       │
       │  BaslerCamera, MockCamera, get_camera
       │  are imported by:
       │
       ├──▶ defect_detector.py
       │       Uses get_camera() in ProductionInspectionSystem
       │       for continuous real-time inspection
       │
       ├──▶ run_pipeline.py
       │       step_validate() uses images captured by this script
       │
       └──▶ validate_images.py
               Validates quality of images saved by this script
               before they're used for training
```

> **The output of this script (`dataset/raw/{label}/`) is the input for `validate_images.py` and `prepare_dataset.py`.**

---

*This document covers every line of `camera_capture.py`. For other scripts, see the corresponding guides in this folder.*

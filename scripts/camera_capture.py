"""
Basler Camera Capture Utility
Captures images from Basler camera for defect detection dataset collection.
"""

import os
import json
import tempfile
from datetime import datetime
from pathlib import Path
from PIL import Image

try:
    from pypylon import pylon
    PYPYLON_AVAILABLE = True
except ImportError:
    PYPYLON_AVAILABLE = False
    print("Warning: pypylon not installed. Install with: pip install pypylon")

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Greyscale conversion (algorithm from greyscale_converter.py)
# ---------------------------------------------------------------------------

def _build_rust_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Return float [0,1] mask: 1.0 = strong rust, 0.0 = clean metal."""
    r = img_bgr[:, :, 2].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    b = img_bgr[:, :, 0].astype(np.float32)
    ratio = r / (g + b + 1.0)
    mean_r = ratio.mean()
    mask = np.clip((ratio - mean_r) / mean_r, 0, 1).astype(np.float32)
    return cv2.GaussianBlur(mask, (7, 7), 2)


def convert_to_greyscale(img_bgr: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """
    Convert a BGR image to greyscale with rust-area darkening (CLAHE enhanced).

    Args:
        img_bgr:  Input image in BGR format.
        strength: Rust-darkening strength 0.0-1.0 (default 0.55).

    Returns:
        BGR-encoded greyscale image with rust areas darkened.
    """
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    grey = clahe.apply(grey)
    rust_mask = _build_rust_mask(img_bgr)
    dark_factor = 1.0 - rust_mask * strength
    result = np.clip(grey.astype(np.float32) * dark_factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def select_save_mode() -> str:
    """
    Ask the user once at startup whether to save images as original or greyscale.

    Returns:
        'original' or 'greyscale'
    """
    print("\n" + "=" * 50)
    print("  IMAGE SAVE MODE")
    print("=" * 50)
    print("  [1] original   - colour image as captured")
    print("  [2] greyscale  - rust-aware greyscale (CLAHE)")
    print("=" * 50)
    while True:
        choice = input("  Choose mode (1/2 or 'original'/'greyscale'): ").strip().lower()
        if choice in ("1", "original"):
            print("  Mode: ORIGINAL\n")
            return "original"
        elif choice in ("2", "greyscale", "grey", "gray"):
            print("  Mode: GREYSCALE\n")
            return "greyscale"
        else:
            print("  Invalid — enter 1 or 2.")


class BaslerCamera:
    """Wrapper class for Basler camera operations"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize Basler camera
         
        Args:
            config_path: Path to system_config.json (optional)
        """
        if not PYPYLON_AVAILABLE:
            raise ImportError("pypylon is not installed. Run: pip install pypylon")
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.camera = None
        self.converter = None
        
    def _load_config(self, config_path: str) -> dict:
        """Load camera configuration from JSON file"""
        default_config = {
            "exposure_time_us": 15000,
            "gain_db": 0,
            "pixel_format": "Mono8",
            "trigger_mode": "Software",
            "timeout_ms": 5000
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                full_config = json.load(f)
                return full_config.get('camera', default_config)
        
        return default_config
        
    def connect(self) -> bool:
        """Connect to the first available Basler camera"""
        try:
            # Get the first available camera
            self.camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice()
            )
            self.camera.Open()
            
            # Apply settings
            self._apply_settings()
            
            # Create image converter for consistent output
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            
            print(f"Connected to camera: {self.camera.GetDeviceInfo().GetModelName()}")
            return True
            
        except Exception as e:
            print(f"Failed to connect to camera: {e}")
            return False
    
    def _apply_settings(self):
        """Apply configuration settings to camera"""
        try:
            # Exposure time (microseconds)
            if self.camera.ExposureTime.IsWritable():
                self.camera.ExposureTime.SetValue(self.config['exposure_time_us'])
            
            # Gain (dB)
            if self.camera.Gain.IsWritable():
                self.camera.Gain.SetValue(self.config['gain_db'])
            
            print(f"Camera settings applied:")
            print(f"  - Exposure: {self.config['exposure_time_us']} µs")
            print(f"  - Gain: {self.config['gain_db']} dB")
            
        except Exception as e:
            print(f"Warning: Could not apply all settings: {e}")
    
    def capture(self) -> np.ndarray:
        """
        Capture a single image from the camera
        
        Returns:
            numpy.ndarray: Captured image in BGR format, or None if failed
        """
        if self.camera is None:
            print("Error: Camera not connected")
            return None
        
        try:
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            
            grab_result = self.camera.RetrieveResult(
                self.config['timeout_ms'],
                pylon.TimeoutHandling_ThrowException
            )
            
            if grab_result.GrabSucceeded():
                # Convert to OpenCV format
                image = self.converter.Convert(grab_result)
                img_array = image.GetArray()
                
                grab_result.Release()
                self.camera.StopGrabbing()
                
                return img_array
            else:
                print(f"Grab failed: {grab_result.ErrorCode}")
                grab_result.Release()
                self.camera.StopGrabbing()
                return None
                
        except Exception as e:
            print(f"Capture error: {e}")
            return None
    
    def capture_and_save(self, output_dir: str, label: str, 
                         metadata: dict = None) -> str:
        """
        Capture image and save to disk with proper naming
        
        Args:
            output_dir: Base output directory
            label: Class label (e.g., 'good', 'scratch', 'crack')
            metadata: Optional metadata to include in filename
            
        Returns:
            str: Path to saved image, or None if failed
        """
        image = self.capture()
        
        if image is None:
            return None
        
        # Create output directory
        save_dir = Path(output_dir) / label
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{label}_{timestamp}.png"
        filepath = save_dir / filename
        
        # Save image (PNG for lossless quality)
        cv2.imwrite(str(filepath), image)
        
        print(f"Saved: {filepath}")
        return str(filepath)
    
    def disconnect(self):
        """Disconnect from camera"""
        if self.camera is not None:
            self.camera.Close()
            self.camera = None
            print("Camera disconnected")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


class MockCamera:
    """Mock camera for testing without hardware"""
    
    def __init__(self, config_path: str = None):
        self.config = {"exposure_time_us": 15000, "gain_db": 0}
        print("Using MockCamera (no hardware connected)")
    
    def connect(self) -> bool:
        print("MockCamera connected (simulated)")
        return True
    
    def capture(self) -> np.ndarray:
        """Generate a test pattern image"""
        # Create a 640x480 test image with random noise
        image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        
        # Add some simulated "defects" (random dark spots)
        if np.random.random() > 0.5:
            x, y = np.random.randint(50, 590), np.random.randint(50, 430)
            cv2.rectangle(image, (x, y), (x+50, y+20), (50, 50, 50), -1)
        
        return image
    
    def capture_and_save(self, output_dir: str, label: str, 
                         metadata: dict = None) -> str:
        image = self.capture()
        
        save_dir = Path(output_dir) / label
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{label}_{timestamp}.png"
        filepath = save_dir / filename
        
        cv2.imwrite(str(filepath), image)
        print(f"Saved (mock): {filepath}")
        return str(filepath)
    
    def disconnect(self):
        print("MockCamera disconnected")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def get_camera(config_path: str = None, use_mock: bool = False):
    """
    Factory function to get appropriate camera instance
    
    Args:
        config_path: Path to configuration file
        use_mock: Force use of mock camera
        
    Returns:
        Camera instance (BaslerCamera or MockCamera)
    """
    if use_mock or not PYPYLON_AVAILABLE:
        return MockCamera(config_path)
    return BaslerCamera(config_path)


def preview_image(image: np.ndarray, title: str = "Captured Image Preview") -> bool:
    """
    Display a preview of the captured image using the default system viewer.
    
    Args:
        image: The captured image as a numpy array (BGR format from OpenCV)
        title: Window title for the preview
        
    Returns:
        True if the user accepts the image, False if retake is requested
    """
    # Convert BGR (OpenCV) to RGB (Pillow)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Resize for preview if the image is very large
    max_preview_size = 800
    if max(pil_image.size) > max_preview_size:
        pil_image.thumbnail((max_preview_size, max_preview_size))
    
    # Save to a temp file and open with default viewer
    temp_path = os.path.join(tempfile.gettempdir(), "_camera_preview.png")
    pil_image.save(temp_path)
    os.startfile(temp_path)  # Opens with default Windows photo viewer
    
    print(f"\n  [PREVIEW] '{title}' opened in your default image viewer.")
    choice = input("  Type 'r' to RETAKE, or press Enter to ACCEPT: ").strip().lower()
    
    if choice == 'r':
        print("  ↻ Retake requested.")
        return False
    
    print("  ✓ Image accepted.")
    return True


# Interactive collection mode
def collect_dataset_interactive():
    """
    Interactive mode for collecting dataset images.
    Asks once at startup whether to save as original colour or greyscale.
    """
    # --- Ask for save mode before anything else ---
    save_mode = select_save_mode()

    print("\n" + "="*50)
    print("DATASET COLLECTION MODE")
    print("="*50)
    print(f"Save mode : {save_mode.upper()}")
    print("Keys      : g=Good  s=Scratch  c=Crack  d=Dent  r=Rust  y=Other  q=Quit")
    print("-"*50)
    print("After each capture a PREVIEW window will open.")
    print("Type 'r' to retake, or press Enter to accept & save.")
    print("="*50 + "\n")

    output_dir = "dataset/raw"

    camera = get_camera(use_mock=not PYPYLON_AVAILABLE)

    with camera:
        while True:
            key = input("\nEnter label key (g/s/c/d/r/y) or q to quit: ").lower().strip()

            label_map = {
                'g': 'good',
                's': 'scratch',
                'c': 'crack',
                'd': 'dent',
                'r': 'rust',
                'y': 'other'
            }

            if key == 'q':
                print("Exiting collection mode...")
                break
            elif key in label_map:
                label = label_map[key]

                # Capture → preview → retake loop
                while True:
                    image = camera.capture()
                    if image is None:
                        print("✗ Capture failed. Try again.")
                        break

                    accepted = preview_image(image, title=f"Preview — {label} [{save_mode}]")

                    if accepted:
                        save_dir = Path(output_dir) / label
                        save_dir.mkdir(parents=True, exist_ok=True)

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"{label}_{timestamp}.png"
                        filepath = save_dir / filename

                        # Apply greyscale conversion if requested
                        if save_mode == "greyscale":
                            image_to_save = convert_to_greyscale(image)
                            print(f"  [greyscale] Converting before save...")
                        else:
                            image_to_save = image

                        cv2.imwrite(str(filepath), image_to_save)
                        print(f"  ✓ Saved {label} ({save_mode}) → {filepath}")
                        break
                    else:
                        print("  Retaking image...")
            else:
                print("Invalid key. Use g/s/c/d/r/y or q to quit.")


if __name__ == "__main__":
    # Run interactive collection mode
    collect_dataset_interactive()

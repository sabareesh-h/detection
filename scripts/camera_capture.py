"""
Basler Camera Capture Utility
Captures images from Basler camera for defect detection dataset collection.
"""

import os
import json
from datetime import datetime
from pathlib import Path

try:
    from pypylon import pylon
    PYPYLON_AVAILABLE = True
except ImportError:
    PYPYLON_AVAILABLE = False
    print("Warning: pypylon not installed. Install with: pip install pypylon")

import cv2
import numpy as np


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


# Interactive collection mode
def collect_dataset_interactive():
    """
    Interactive mode for collecting dataset images
    Press keys to capture images with different labels
    """
    print("\n" + "="*50)
    print("DATASET COLLECTION MODE")
    print("="*50)
    print("Press keys to capture images:")
    print("  g - Good product")
    print("  s - Scratch defect")
    print("  c - Crack defect")
    print("  d - Dent defect")
    print("  x - Discoloration/Other defect")
    print("  q - Quit")
    print("="*50 + "\n")
    
    output_dir = "dataset/raw"
    
    camera = get_camera(use_mock=not PYPYLON_AVAILABLE)
    
    with camera:
        while True:
            key = input("\nEnter label key (g/s/c/d/x) or q to quit: ").lower().strip()
            
            label_map = {
                'g': 'good',
                's': 'scratch',
                'c': 'crack',
                'd': 'dent',
                'x': 'other'
            }
            
            if key == 'q':
                print("Exiting collection mode...")
                break
            elif key in label_map:
                label = label_map[key]
                filepath = camera.capture_and_save(output_dir, label)
                if filepath:
                    print(f"✓ Captured {label} image")
            else:
                print("Invalid key. Use g/s/c/d/x or q to quit.")


if __name__ == "__main__":
    # Run interactive collection mode
    collect_dataset_interactive()

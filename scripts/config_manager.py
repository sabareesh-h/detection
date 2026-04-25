import json
import os
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator

class CameraConfig(BaseModel):
    exposure_time_us: int = Field(default=15000, ge=100)
    gain_db: int = Field(default=0, ge=0)
    pixel_format: str = Field(default="Mono8")
    trigger_mode: str = Field(default="Software")
    timeout_ms: int = Field(default=5000, ge=1000)

class ModelConfig(BaseModel):
    weights_path: str = Field(default="models/best.pt")
    confidence_threshold: float = Field(default=0.03, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    image_size: int = Field(default=640, ge=32)
    gpu_preprocessing: bool = Field(default=False)

class InspectionConfig(BaseModel):
    save_images: bool = Field(default=True)
    save_path: str = Field(default="logs/inspections")
    log_to_database: bool = Field(default=True)
    database_path: str = Field(default="logs/inspections.db")
    min_defect_area_px: float = Field(default=150.0, ge=0.0)
    dynamic_margin: float = Field(default=0.15, ge=0.0, le=1.0)
    roi: Optional[List[int]] = Field(default=None)

    @field_validator('roi')
    @classmethod
    def check_roi(cls, v):
        if v is not None and len(v) != 4:
            raise ValueError("ROI must be a list of 4 integers: [ymin, ymax, xmin, xmax]")
        return v

class SystemConfig(BaseModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    inspection: InspectionConfig = Field(default_factory=InspectionConfig)
    classes: List[str] = Field(default_factory=lambda: ["Scratch", "Rust"])

def load_system_config(config_path: str) -> SystemConfig:
    """
    Load and validate the system configuration from a JSON file.
    If the file doesn't exist, returns the default SystemConfig.
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data = json.load(f)
            # Pydantic v2 validation from dict
            return SystemConfig(**data)
    
    print(f"Warning: Configuration file {config_path} not found. Using default settings.")
    return SystemConfig()

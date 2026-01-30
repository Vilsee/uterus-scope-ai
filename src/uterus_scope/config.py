"""
Configuration management for UterusScope-AI.

Handles environment variables, defaults, and validation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class ModelBackbone(str, Enum):
    """Supported model backbones."""
    SWIN_TINY = "swin_tiny"
    SWIN_SMALL = "swin_small"
    SWIN_BASE = "swin_base"


class PretrainedWeights(str, Enum):
    """Pretrained weight options."""
    IMAGENET = "imagenet"
    RADIMAGENET = "radimagenet"
    NONE = "none"


class Device(str, Enum):
    """Device options for inference."""
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"


class ReportFormat(str, Enum):
    """Report output formats."""
    PDF = "pdf"
    HTML = "html"


class VascularityType(int, Enum):
    """Vascularity classification types."""
    TYPE_0 = 0  # Avascular
    TYPE_I = 1  # Minimal flow
    TYPE_II = 2  # Moderate flow
    TYPE_III = 3  # High vascularity


class CandidacyStatus(str, Enum):
    """UG-IHI candidacy classification."""
    EXCELLENT_CANDIDATE = "excellent_candidate"
    GOOD_CANDIDATE = "good_candidate"
    CAUTIONARY = "cautionary"
    NOT_RECOMMENDED = "not_recommended"


@dataclass
class ModelConfig:
    """Model configuration settings."""
    backbone: ModelBackbone = ModelBackbone.SWIN_TINY
    pretrained_weights: PretrainedWeights = PretrainedWeights.IMAGENET
    device: Device = Device.CUDA
    checkpoint_path: Optional[Path] = None
    input_size: int = 224
    
    @classmethod
    def from_env(cls) -> ModelConfig:
        """Create configuration from environment variables."""
        return cls(
            backbone=ModelBackbone(os.getenv("MODEL_BACKBONE", "swin_tiny")),
            pretrained_weights=PretrainedWeights(os.getenv("PRETRAINED_WEIGHTS", "imagenet")),
            device=Device(os.getenv("DEVICE", "cuda")),
            checkpoint_path=Path(p) if (p := os.getenv("MODEL_CHECKPOINT_PATH")) else None,
            input_size=int(os.getenv("INPUT_SIZE", "224")),
        )


@dataclass
class ProcessingConfig:
    """Data processing configuration."""
    video_fps: int = 5
    max_frames: int = 100
    batch_size: int = 8
    
    @classmethod
    def from_env(cls) -> ProcessingConfig:
        """Create configuration from environment variables."""
        return cls(
            video_fps=int(os.getenv("VIDEO_FPS", "5")),
            max_frames=int(os.getenv("MAX_FRAMES", "100")),
            batch_size=int(os.getenv("BATCH_SIZE", "8")),
        )


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    debug: bool = True
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])
    
    @classmethod
    def from_env(cls) -> APIConfig:
        """Create configuration from environment variables."""
        cors = os.getenv("CORS_ORIGINS", "http://localhost:3000")
        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            workers=int(os.getenv("API_WORKERS", "4")),
            debug=os.getenv("API_DEBUG", "true").lower() == "true",
            cors_origins=[o.strip() for o in cors.split(",")],
        )


@dataclass
class ReportConfig:
    """Report generation configuration."""
    output_dir: Path = field(default_factory=lambda: Path("./reports"))
    default_format: ReportFormat = ReportFormat.PDF
    include_heatmaps: bool = True
    
    @classmethod
    def from_env(cls) -> ReportConfig:
        """Create configuration from environment variables."""
        return cls(
            output_dir=Path(os.getenv("REPORT_OUTPUT_DIR", "./reports")),
            default_format=ReportFormat(os.getenv("DEFAULT_REPORT_FORMAT", "pdf")),
            include_heatmaps=os.getenv("INCLUDE_HEATMAPS", "true").lower() == "true",
        )


@dataclass
class ClinicalThresholds:
    """Clinical threshold configuration."""
    # Endometrial thickness (mm)
    thickness_min_normal: float = 5.0
    thickness_max_normal: float = 10.0
    thickness_alert_high: float = 15.0
    
    # Vascularity
    vascularity_alert_threshold: int = 3
    
    # Fibrosis score
    fibrosis_normal_max: float = 0.3
    fibrosis_alert_threshold: float = 0.5
    
    @classmethod
    def from_env(cls) -> ClinicalThresholds:
        """Create configuration from environment variables."""
        return cls(
            thickness_min_normal=float(os.getenv("THICKNESS_MIN_NORMAL", "5.0")),
            thickness_max_normal=float(os.getenv("THICKNESS_MAX_NORMAL", "10.0")),
            thickness_alert_high=float(os.getenv("THICKNESS_ALERT_HIGH", "15.0")),
            vascularity_alert_threshold=int(os.getenv("VASCULARITY_ALERT_THRESHOLD", "3")),
            fibrosis_normal_max=float(os.getenv("FIBROSIS_NORMAL_MAX", "0.3")),
            fibrosis_alert_threshold=float(os.getenv("FIBROSIS_ALERT_THRESHOLD", "0.5")),
        )


@dataclass
class Config:
    """Main configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    clinical: ClinicalThresholds = field(default_factory=ClinicalThresholds)
    
    # Paths
    synthetic_data_dir: Path = field(default_factory=lambda: Path("./data/synthetic"))
    model_cache_dir: Path = field(default_factory=lambda: Path("./cache/models"))
    
    # Logging
    log_level: str = "INFO"
    log_file_path: Optional[Path] = None
    
    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> Config:
        """
        Create configuration from environment variables.
        
        Args:
            env_file: Optional path to .env file
            
        Returns:
            Config instance with all settings
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        log_path = os.getenv("LOG_FILE_PATH")
        
        return cls(
            model=ModelConfig.from_env(),
            processing=ProcessingConfig.from_env(),
            api=APIConfig.from_env(),
            report=ReportConfig.from_env(),
            clinical=ClinicalThresholds.from_env(),
            synthetic_data_dir=Path(os.getenv("SYNTHETIC_DATA_DIR", "./data/synthetic")),
            model_cache_dir=Path(os.getenv("MODEL_CACHE_DIR", "./cache/models")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file_path=Path(log_path) if log_path else None,
        )
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.report.output_dir.mkdir(parents=True, exist_ok=True)
        self.synthetic_data_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config

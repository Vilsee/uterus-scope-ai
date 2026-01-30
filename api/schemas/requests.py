"""Request schemas for API."""

from pydantic import BaseModel, Field
from typing import Optional


class AnalysisRequest(BaseModel):
    """Request for analysis."""
    include_heatmaps: bool = Field(default=True, description="Include explainability heatmaps")
    include_report: bool = Field(default=False, description="Generate report immediately")


class FrameAnalysisRequest(BaseModel):
    """Request for single frame analysis."""
    frame_data: str = Field(..., description="Base64 encoded frame data")
    include_heatmaps: bool = True


class VideoAnalysisRequest(BaseModel):
    """Request for video analysis."""
    video_url: Optional[str] = Field(None, description="URL to video file")
    max_frames: int = Field(default=100, description="Maximum frames to analyze")
    fps: int = Field(default=5, description="Frames per second to extract")

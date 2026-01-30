"""Response schemas for API."""

from pydantic import BaseModel, Field
from typing import Optional


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")


class AnalysisResponse(BaseModel):
    """Analysis result response."""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    endometrial_thickness_mm: float = Field(..., description="Measured thickness in mm")
    vascularity_type: int = Field(..., description="Vascularity classification 0-3")
    vascularity_confidence: float = Field(..., description="Vascularity prediction confidence")
    fibrosis_score: float = Field(..., description="Fibrosis severity score 0-1")
    candidacy: str = Field(..., description="UG-IHI candidacy status")
    candidacy_confidence: float = Field(..., description="Candidacy assessment confidence")
    summary: str = Field(..., description="Clinical summary")
    alerts: list[str] = Field(default=[], description="Clinical alerts")


class ReportResponse(BaseModel):
    """Report generation response."""
    report_url: str = Field(..., description="URL to download report")
    format: str = Field(..., description="Report format (html/pdf)")


class HeatmapResponse(BaseModel):
    """Heatmap response."""
    heatmaps: dict[str, str] = Field(..., description="Base64 encoded heatmaps")

"""
UterusScope-AI Python SDK Client.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import httpx

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result from ultrasound analysis."""
    analysis_id: str
    endometrial_thickness: float
    vascularity_type: int
    vascularity_confidence: float
    fibrosis_score: float
    candidacy: str
    candidacy_confidence: float
    summary: str
    alerts: list[str]
    
    @property
    def vascularity_name(self) -> str:
        names = {0: "Type 0 (Avascular)", 1: "Type I (Minimal)", 
                 2: "Type II (Moderate)", 3: "Type III (High)"}
        return names.get(self.vascularity_type, f"Type {self.vascularity_type}")


class UterusScopeClient:
    """
    Python SDK client for UterusScope-AI API.
    
    Usage:
        client = UterusScopeClient(api_url="http://localhost:8000")
        result = client.analyze_frame("ultrasound.png")
        print(f"Thickness: {result.endometrial_thickness}mm")
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        timeout: float = 60.0,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the client.
        
        Args:
            api_url: Base URL of the API
            timeout: Request timeout in seconds
            api_key: Optional API key for authentication
        """
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make HTTP request."""
        url = f"{self.api_url}{endpoint}"
        
        with httpx.Client(timeout=self.timeout) as client:
            response = client.request(method, url, headers=self.headers, **kwargs)
            response.raise_for_status()
            return response.json()
    
    def health_check(self) -> dict:
        """Check API health."""
        return self._request("GET", "/api/v1/health")
    
    def analyze_frame(self, image_path: Union[str, Path]) -> AnalysisResult:
        """
        Analyze a single ultrasound frame.
        
        Args:
            image_path: Path to image file
            
        Returns:
            AnalysisResult with measurements and recommendations
        """
        image_path = Path(image_path)
        
        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/png")}
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.api_url}/api/v1/analyze/frame",
                    files=files,
                    headers=self.headers,
                )
                response.raise_for_status()
                data = response.json()
        
        return AnalysisResult(
            analysis_id=data["analysis_id"],
            endometrial_thickness=data["endometrial_thickness_mm"],
            vascularity_type=data["vascularity_type"],
            vascularity_confidence=data["vascularity_confidence"],
            fibrosis_score=data["fibrosis_score"],
            candidacy=data["candidacy"],
            candidacy_confidence=data["candidacy_confidence"],
            summary=data["summary"],
            alerts=data.get("alerts", []),
        )
    
    def analyze_video(self, video_path: Union[str, Path]) -> AnalysisResult:
        """
        Analyze ultrasound video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            AnalysisResult with temporally aggregated measurements
        """
        video_path = Path(video_path)
        
        with open(video_path, "rb") as f:
            files = {"file": (video_path.name, f, "video/mp4")}
            
            with httpx.Client(timeout=self.timeout * 3) as client:
                response = client.post(
                    f"{self.api_url}/api/v1/analyze/video",
                    files=files,
                    headers=self.headers,
                )
                response.raise_for_status()
                data = response.json()
        
        return AnalysisResult(
            analysis_id=data["analysis_id"],
            endometrial_thickness=data["endometrial_thickness_mm"],
            vascularity_type=data["vascularity_type"],
            vascularity_confidence=data["vascularity_confidence"],
            fibrosis_score=data["fibrosis_score"],
            candidacy=data["candidacy"],
            candidacy_confidence=data["candidacy_confidence"],
            summary=data["summary"],
            alerts=data.get("alerts", []),
        )
    
    def generate_report(
        self,
        analysis_id: str,
        format: str = "html",
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Generate and download clinical report.
        
        Args:
            analysis_id: ID from previous analysis
            format: Report format ('html' or 'pdf')
            output_path: Optional path to save report
            
        Returns:
            Path to saved report file
        """
        url = f"{self.api_url}/api/v1/reports/{analysis_id}?format={format}"
        
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, headers=self.headers)
            response.raise_for_status()
            
            if output_path is None:
                output_path = Path(f"{analysis_id}_report.{format}")
            else:
                output_path = Path(output_path)
            
            output_path.write_bytes(response.content)
        
        return output_path
    
    def get_heatmaps(self, analysis_id: str) -> dict[str, bytes]:
        """
        Get explainability heatmaps.
        
        Args:
            analysis_id: ID from previous analysis
            
        Returns:
            Dictionary of heatmap name to image bytes
        """
        import base64
        
        url = f"{self.api_url}/api/v1/explain/{analysis_id}"
        
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
        
        heatmaps = {}
        for name, b64_data in data.get("heatmaps", {}).items():
            heatmaps[name] = base64.b64decode(b64_data)
        
        return heatmaps

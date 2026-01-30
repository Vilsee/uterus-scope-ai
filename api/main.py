"""
FastAPI application for UterusScope-AI.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from uterus_scope.config import get_config
from uterus_scope.models.unified import UterusScopeModel
from uterus_scope.agents.decision import ClinicalDecisionAgent
from uterus_scope.data.preprocessing import UltrasoundPreprocessor
from uterus_scope.explainability.gradcam import VisionTransformerGradCAM
from uterus_scope.reports.generator import ClinicalReportGenerator

from api.schemas.requests import AnalysisRequest, FrameAnalysisRequest
from api.schemas.responses import AnalysisResponse, HealthResponse, ReportResponse

logger = logging.getLogger(__name__)

# Global model instance
model: Optional[UterusScopeModel] = None
preprocessor: Optional[UltrasoundPreprocessor] = None
decision_agent: Optional[ClinicalDecisionAgent] = None
report_generator: Optional[ClinicalReportGenerator] = None
gradcam: Optional[VisionTransformerGradCAM] = None

# Analysis cache
analysis_cache: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global model, preprocessor, decision_agent, report_generator, gradcam
    
    config = get_config()
    device = config.model.device.value
    
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
        logger.warning("CUDA not available, using CPU")
    
    logger.info(f"Loading model on {device}...")
    
    # Initialize components
    model = UterusScopeModel(pretrained=True).to(device)
    model.eval()
    
    preprocessor = UltrasoundPreprocessor(device=device)
    decision_agent = ClinicalDecisionAgent()
    report_generator = ClinicalReportGenerator()
    gradcam = VisionTransformerGradCAM(model, device=device)
    
    logger.info("Model loaded successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")


app = FastAPI(
    title="UterusScope-AI API",
    description="Autonomous ultrasound interpretation for endometrial analysis",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API info."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model_loaded=model is not None,
    )


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model_loaded=model is not None,
    )


@app.post("/api/v1/analyze/frame", response_model=AnalysisResponse)
async def analyze_frame(file: UploadFile = File(...)):
    """Analyze a single ultrasound frame."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and preprocess image
        import cv2
        import numpy as np
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Preprocess
        tensor = preprocessor.preprocess_frame(image).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            output = model(tensor)
        
        # Clinical decision
        decision = decision_agent.evaluate(output)
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())[:8]
        
        # Cache results
        analysis_cache[analysis_id] = {
            'output': output,
            'decision': decision,
            'image': image,
        }
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            endometrial_thickness_mm=float(output.segmentation.thickness_mm[0]),
            vascularity_type=int(output.vascularity.predicted_type[0]),
            vascularity_confidence=float(output.vascularity.confidence[0]),
            fibrosis_score=float(output.fibrosis.severity_score[0]),
            candidacy=decision.candidacy.value,
            candidacy_confidence=decision.candidacy_confidence,
            summary=decision.summary,
            alerts=decision.alerts,
        )
    
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze/video", response_model=AnalysisResponse)
async def analyze_video(file: UploadFile = File(...)):
    """Analyze ultrasound video."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import tempfile
        
        # Save video temporarily
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        # Preprocess video
        frames_tensor = preprocessor.preprocess_video(tmp_path)
        
        # Cleanup temp file
        Path(tmp_path).unlink()
        
        # Run video inference
        with torch.no_grad():
            output = model.forward_video(frames_tensor.unsqueeze(0))
        
        # Clinical decision
        decision = decision_agent.evaluate(output)
        
        analysis_id = str(uuid.uuid4())[:8]
        analysis_cache[analysis_id] = {'output': output, 'decision': decision}
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            endometrial_thickness_mm=float(output.segmentation.thickness_mm[0]),
            vascularity_type=int(output.vascularity.predicted_type[0]),
            vascularity_confidence=float(output.vascularity.confidence[0]),
            fibrosis_score=float(output.fibrosis.severity_score[0]),
            candidacy=decision.candidacy.value,
            candidacy_confidence=decision.candidacy_confidence,
            summary=decision.summary,
            alerts=decision.alerts,
        )
    
    except Exception as e:
        logger.exception("Video analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/reports/{analysis_id}")
async def get_report(analysis_id: str, format: str = "html"):
    """Get analysis report."""
    if analysis_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    cached = analysis_cache[analysis_id]
    
    from uterus_scope.config import ReportFormat
    report_format = ReportFormat.PDF if format.lower() == "pdf" else ReportFormat.HTML
    
    # Generate heatmaps if available
    heatmaps = None
    if 'image' in cached and gradcam is not None:
        try:
            tensor = preprocessor.preprocess_frame(cached['image']).unsqueeze(0)
            heatmaps = gradcam.generate_multi_target(tensor)
        except Exception as e:
            logger.warning(f"Failed to generate heatmaps: {e}")
    
    report_path = report_generator.generate(
        cached['output'],
        cached['decision'],
        analysis_id,
        heatmaps=heatmaps,
        format=report_format,
    )
    
    return FileResponse(
        path=str(report_path),
        filename=report_path.name,
        media_type="text/html" if format == "html" else "application/pdf",
    )


@app.post("/api/v1/explain/{analysis_id}")
async def get_explainability(analysis_id: str):
    """Get explainability heatmaps."""
    if analysis_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if gradcam is None:
        raise HTTPException(status_code=503, detail="Explainability not available")
    
    cached = analysis_cache[analysis_id]
    
    if 'image' not in cached:
        raise HTTPException(status_code=400, detail="Original image not available")
    
    tensor = preprocessor.preprocess_frame(cached['image']).unsqueeze(0)
    heatmaps = gradcam.generate_multi_target(tensor)
    
    # Convert to base64 for JSON response
    import base64
    import cv2
    
    encoded = {}
    for name, hm in heatmaps.items():
        hm_uint8 = (hm * 255).astype('uint8')
        _, buffer = cv2.imencode('.png', hm_uint8)
        encoded[name] = base64.b64encode(buffer).decode('utf-8')
    
    return JSONResponse(content={"heatmaps": encoded})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.api.host, port=config.api.port)

# 🔬 UterusScope-AI

**Autonomous Ultrasound Interpretation Agent for Endometrial Analysis**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MONAI](https://img.shields.io/badge/MONAI-1.3+-green.svg)](https://monai.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)](https://fastapi.tiangolo.com/)

> ⚠️ **Medical Disclaimer**: This system is a clinical decision support tool designed to assist healthcare professionals. It is NOT intended to replace professional medical judgment. All outputs require review by qualified clinicians before any clinical decisions are made.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Processing Pipeline](#-processing-pipeline)
- [AI Models](#-ai-models)
- [Clinical Decision Flow](#-clinical-decision-flow)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Clinical Thresholds](#-clinical-thresholds)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [License & Disclaimer](#-license--disclaimer)

---

## 🎯 Overview

UterusScope-AI is an open-source autonomous agent that analyzes transvaginal ultrasound videos to assess:

| Analysis | Description | Output |
|----------|-------------|--------|
| 📏 **Endometrial Thickness** | Precise measurement with segmentation | Millimeters (mm) |
| 🩸 **Vascularity Patterns** | Blood flow classification | Types 0-III |
| 🔬 **Fibrosis Risk** | Scar pattern detection | Risk score (0-1) |

The system assists in **UG-IHI (Uterine Gel-based Intrauterine Hydrogel Infusion)** candidacy decisions by providing **explainable, clinician-readable reports** with GradCAM heatmaps.

---

## ✨ Key Features

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         UterusScope-AI Features                          │
├──────────────────┬──────────────────┬───────────────────┬───────────────┤
│   🧠 AI Models   │  📊 Explainability │   🔌 Integration  │  📋 Reports  │
├──────────────────┼──────────────────┼───────────────────┼───────────────┤
│ Swin Transformer │ GradCAM++ Maps   │ REST API          │ HTML/PDF      │
│ Multi-task Heads │ Attention Viz    │ Python SDK        │ Heatmaps      │
│ Video Temporal   │ Rollout Analysis │ ONNX Export       │ Candidacy     │
│ MONAI Pipeline   │ Per-head Explain │ CORS Support      │ Risk Scores   │
└──────────────────┴──────────────────┴───────────────────┴───────────────┘
```

---

## 🏗️ System Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph Input["📥 Input Layer"]
        US[("🔊 Ultrasound<br/>Video/Frame")]
    end

    subgraph Preprocessing["🔧 Preprocessing"]
        FE["Frame<br/>Extraction"]
        CLAHE["CLAHE<br/>Enhancement"]
        DN["Speckle<br/>Denoising"]
        NORM["Normalization<br/>& Resize"]
    end

    subgraph Backbone["🧠 Vision Transformer"]
        SWIN["Swin Transformer<br/>(Tiny/Small/Base)"]
        FPN["Feature Pyramid<br/>Network"]
    end

    subgraph Heads["🎯 Analysis Heads"]
        SEG["Segmentation<br/>Head"]
        VASC["Vascularity<br/>Classifier"]
        FIB["Fibrosis<br/>Detector"]
        TEMP["Temporal<br/>Aggregator"]
    end

    subgraph Agents["🤖 Clinical Agents"]
        DEC["Decision<br/>Agent"]
        RISK["Risk<br/>Scorer"]
        CAND["Candidacy<br/>Assessor"]
    end

    subgraph Output["📤 Output Layer"]
        REP["Clinical<br/>Report"]
        HEAT["GradCAM<br/>Heatmaps"]
        API["REST API<br/>Response"]
    end

    US --> FE --> CLAHE --> DN --> NORM
    NORM --> SWIN --> FPN
    FPN --> SEG & VASC & FIB
    SEG & VASC & FIB --> TEMP
    TEMP --> DEC --> RISK --> CAND
    CAND --> REP & HEAT & API

    style US fill:#e1f5fe
    style SWIN fill:#fff3e0
    style DEC fill:#e8f5e9
    style REP fill:#fce4ec
```

### Component Interaction Diagram

```mermaid
graph LR
    subgraph Client["👤 Client"]
        SDK["Python SDK"]
        HTTP["HTTP Client"]
    end

    subgraph API["🌐 FastAPI Server"]
        FRAME["/analyze/frame"]
        VIDEO["/analyze/video"]
        REPORT["/reports/{id}"]
        EXPLAIN["/explain/{id}"]
    end

    subgraph Core["⚙️ Core Engine"]
        MODEL["Unified Model"]
        AGENT["Decision Agent"]
        GEN["Report Generator"]
    end

    subgraph Storage["💾 Storage"]
        CACHE["Analysis Cache"]
        FILES["Report Files"]
    end

    SDK --> HTTP --> FRAME & VIDEO
    FRAME & VIDEO --> MODEL --> AGENT --> CACHE
    REPORT --> GEN --> FILES
    EXPLAIN --> MODEL

    style SDK fill:#bbdefb
    style MODEL fill:#fff9c4
    style CACHE fill:#c8e6c9
```

---

## 🔄 Processing Pipeline

### Frame Processing Workflow

```mermaid
flowchart LR
    subgraph Input
        RAW["Raw Frame<br/>(BGR)"]
    end

    subgraph Enhancement["Image Enhancement"]
        GRAY["Grayscale<br/>Conversion"]
        CLAHE["CLAHE<br/>Contrast"]
        SPECKLE["Speckle<br/>Reduction"]
    end

    subgraph Normalization
        RESIZE["Resize<br/>(224×224)"]
        NORM["Min-Max<br/>Normalize"]
        TENSOR["PyTorch<br/>Tensor"]
    end

    RAW --> GRAY --> CLAHE --> SPECKLE --> RESIZE --> NORM --> TENSOR

    style RAW fill:#ffcdd2
    style TENSOR fill:#c8e6c9
```

### Video Analysis Pipeline

```mermaid
flowchart TB
    VIDEO["📹 Input Video"] --> EXTRACT["Extract Frames<br/>(N fps)"]
    EXTRACT --> BATCH["Batch<br/>Preprocessing"]
    
    subgraph FrameAnalysis["Per-Frame Analysis"]
        F1["Frame 1"]
        F2["Frame 2"]
        FN["Frame N"]
    end
    
    BATCH --> F1 & F2 & FN
    
    F1 & F2 & FN --> TEMPORAL["Temporal<br/>Aggregator"]
    
    subgraph Methods["Aggregation Methods"]
        ATT["Attention<br/>Weighted"]
        LSTM["LSTM<br/>Sequential"]
        CONF["Confidence<br/>Weighted"]
    end
    
    TEMPORAL --> ATT & LSTM & CONF --> FINAL["Final<br/>Prediction"]

    style VIDEO fill:#e3f2fd
    style FINAL fill:#e8f5e9
```

---

## 🧠 AI Models

### Model Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           UterusScopeModel                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Swin Transformer Backbone                         │   │
│  │  ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐      │   │
│  │  │  Stage 1  │──▶│  Stage 2  │──▶│  Stage 3  │──▶│  Stage 4  │      │   │
│  │  │  96 ch    │   │  192 ch   │   │  384 ch   │   │  768 ch   │      │   │
│  │  │  56×56    │   │  28×28    │   │  14×14    │   │  7×7      │      │   │
│  │  └───────────┘   └───────────┘   └───────────┘   └───────────┘      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│           ┌────────────────────────┼────────────────────────┐              │
│           │                        │                        │              │
│           ▼                        ▼                        ▼              │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐      │
│  │   Segmentation  │     │   Vascularity   │     │    Fibrosis     │      │
│  │      Head       │     │    Classifier   │     │    Detector     │      │
│  ├─────────────────┤     ├─────────────────┤     ├─────────────────┤      │
│  │ U-Net Decoder   │     │ Attention Pool  │     │ Spatial Attn    │      │
│  │ Thickness Est.  │     │ 4-class Output  │     │ Severity Score  │      │
│  │ Binary Mask     │     │ Confidence      │     │ Probability Map │      │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘      │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Model Specifications

| Component | Architecture | Parameters | Output |
|-----------|--------------|------------|--------|
| **Backbone** | Swin-Tiny | ~28M | Multi-scale features |
| **Segmentation** | U-Net Decoder | ~5M | 224×224 mask + thickness |
| **Vascularity** | Attention Pooling + MLP | ~1M | 4 classes + confidence |
| **Fibrosis** | Spatial Attention + Conv | ~2M | Probability map + score |
| **Temporal** | Multi-head Attention | ~0.5M | Aggregated features |

### Swin Transformer Variants

```
┌─────────────────┬────────────┬─────────────┬────────────┬──────────────┐
│     Variant     │ Embed Dim  │   Depths    │   Heads    │   Params     │
├─────────────────┼────────────┼─────────────┼────────────┼──────────────┤
│   Swin-Tiny     │     96     │ [2,2,6,2]   │ [3,6,12,24]│    ~28M      │
│   Swin-Small    │     96     │ [2,2,18,2]  │ [3,6,12,24]│    ~50M      │
│   Swin-Base     │    128     │ [2,2,18,2]  │ [4,8,16,32]│    ~88M      │
└─────────────────┴────────────┴─────────────┴────────────┴──────────────┘
```

---

## 🏥 Clinical Decision Flow

### Decision Agent Workflow

```mermaid
flowchart TB
    INPUT["Model Output<br/>(Thickness, Vascularity, Fibrosis)"]
    
    subgraph Evaluation["📊 Threshold Evaluation"]
        T_CHECK{"Thickness<br/>5-10mm?"}
        V_CHECK{"Vascularity<br/>< Type III?"}
        F_CHECK{"Fibrosis<br/>< 0.5?"}
    end
    
    subgraph Risk["⚠️ Risk Assessment"]
        LOW["Low Risk<br/>(< 0.2)"]
        MOD["Moderate Risk<br/>(0.2 - 0.4)"]
        HIGH["High Risk<br/>(0.4 - 0.6)"]
        VHIGH["Very High Risk<br/>(> 0.6)"]
    end
    
    subgraph Candidacy["✅ Candidacy Status"]
        EXCELLENT["Excellent<br/>Candidate"]
        GOOD["Good<br/>Candidate"]
        CAUTION["Cautionary"]
        NOT_REC["Not<br/>Recommended"]
    end
    
    INPUT --> T_CHECK & V_CHECK & F_CHECK
    T_CHECK & V_CHECK & F_CHECK --> LOW & MOD & HIGH & VHIGH
    
    LOW --> EXCELLENT
    MOD --> GOOD
    HIGH --> CAUTION
    VHIGH --> NOT_REC

    style EXCELLENT fill:#c8e6c9
    style GOOD fill:#dcedc8
    style CAUTION fill:#fff9c4
    style NOT_REC fill:#ffcdd2
```

### Risk Scoring Formula

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Risk Score Calculation                           │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Total Risk = Σ (Factor Weight × Factor Score)                        │
│                                                                         │
│   ┌────────────────────┬────────────┬─────────────────────────────┐    │
│   │      Factor        │   Weight   │        Score Range          │    │
│   ├────────────────────┼────────────┼─────────────────────────────┤    │
│   │ Thickness Risk     │    25%     │ 0 (normal) → 1 (extreme)    │    │
│   │ Vascularity Risk   │    25%     │ Type × 0.3 (max 0.9)        │    │
│   │ Fibrosis Risk      │    30%     │ Score × 1.5 (max 1.0)       │    │
│   │ Image Quality      │    10%     │ 1 - quality score           │    │
│   │ Patient Factors    │    10%     │ Age, history based          │    │
│   └────────────────────┴────────────┴─────────────────────────────┘    │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/uterus-scope-ai/uterus-scope-ai.git
cd uterus-scope-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package with all dependencies
pip install -e ".[dev]"
```

### Generate Synthetic Test Data

```bash
# Generate 50 synthetic ultrasound samples
python scripts/generate_synthetic.py --count 50 --output ./data/synthetic

# Generate with video sequences
python scripts/generate_synthetic.py --count 20 --videos --frames 30
```

### Start the API Server

```bash
# Development mode with auto-reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using the Python SDK

```python
from uterus_scope import UterusScopeClient

# Initialize client
client = UterusScopeClient(api_url="http://localhost:8000")

# Health check
status = client.health_check()
print(f"API Status: {status['status']}")

# Analyze single frame
result = client.analyze_frame("ultrasound_frame.png")
print(f"""
╔═══════════════════════════════════════════════════════╗
║              UterusScope-AI Analysis                  ║
╠═══════════════════════════════════════════════════════╣
║  Endometrial Thickness: {result.endometrial_thickness:>6.1f} mm                ║
║  Vascularity Type:      {result.vascularity_name:<24} ║
║  Fibrosis Score:        {result.fibrosis_score:>6.2f}                     ║
║  Confidence:            {result.candidacy_confidence:>6.1%}                    ║
╠═══════════════════════════════════════════════════════╣
║  Candidacy: {result.candidacy:<41} ║
╚═══════════════════════════════════════════════════════╝
""")

# Generate PDF report
report_path = client.generate_report(result.analysis_id, format="pdf")
print(f"Report saved to: {report_path}")

# Get explainability heatmaps
heatmaps = client.get_heatmaps(result.analysis_id)
for name, image_bytes in heatmaps.items():
    with open(f"heatmap_{name}.png", "wb") as f:
        f.write(image_bytes)
```

### Direct Model Usage

```python
import torch
from uterus_scope import get_model, get_config
from uterus_scope.data.preprocessing import UltrasoundPreprocessor

# Load model
Model = get_model()
model = Model(pretrained=True)
model.eval()

# Preprocess image
preprocessor = UltrasoundPreprocessor()
tensor = preprocessor.preprocess_frame(image)

# Run inference
with torch.no_grad():
    output = model(tensor.unsqueeze(0))

print(f"Thickness: {output.segmentation.thickness_mm[0]:.1f}mm")
print(f"Vascularity: Type {output.vascularity.predicted_type[0]}")
print(f"Fibrosis: {output.fibrosis.severity_score[0]:.2f}")
```

---

## 🔌 API Reference

### Endpoints Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          API Endpoints                                   │
├──────────────────────────┬─────────┬────────────────────────────────────┤
│        Endpoint          │ Method  │          Description               │
├──────────────────────────┼─────────┼────────────────────────────────────┤
│ /                        │  GET    │ Root info & health                 │
│ /api/v1/health           │  GET    │ Health check with model status     │
│ /api/v1/analyze/frame    │  POST   │ Analyze single ultrasound frame    │
│ /api/v1/analyze/video    │  POST   │ Analyze ultrasound video           │
│ /api/v1/reports/{id}     │  GET    │ Generate HTML/PDF report           │
│ /api/v1/explain/{id}     │  POST   │ Get GradCAM heatmaps               │
└──────────────────────────┴─────────┴────────────────────────────────────┘
```

### Request/Response Examples

#### Analyze Frame

```bash
curl -X POST "http://localhost:8000/api/v1/analyze/frame" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@ultrasound.png"
```

**Response:**
```json
{
  "analysis_id": "a1b2c3d4",
  "endometrial_thickness_mm": 7.5,
  "vascularity_type": 1,
  "vascularity_confidence": 0.92,
  "fibrosis_score": 0.15,
  "candidacy": "excellent_candidate",
  "candidacy_confidence": 0.89,
  "summary": "Patient appears to be an excellent candidate for UG-IHI therapy.",
  "alerts": []
}
```

#### Get Report

```bash
curl "http://localhost:8000/api/v1/reports/a1b2c3d4?format=pdf" \
  --output report.pdf
```

---

## 📁 Project Structure

```
uterus-scope-ai/
├── 📄 pyproject.toml              # Project configuration & dependencies
├── 📄 README.md                   # This documentation
├── 📄 .env.example               # Environment configuration template
│
├── 📁 api/                        # FastAPI REST API
│   ├── 📄 main.py                # Application & endpoints
│   └── 📁 schemas/               # Pydantic models
│       ├── 📄 requests.py        # Request schemas
│       └── 📄 responses.py       # Response schemas
│
├── 📁 scripts/                    # Utility scripts
│   ├── 📄 generate_synthetic.py  # Synthetic data generation
│   └── 📄 export_onnx.py         # ONNX model export
│
├── 📁 src/uterus_scope/           # Main Python package
│   ├── 📄 __init__.py            # Package exports
│   ├── 📄 config.py              # Configuration management
│   │
│   ├── 📁 data/                  # Data pipeline
│   │   ├── 📄 preprocessing.py   # CLAHE, denoising, normalization
│   │   ├── 📄 synthetic.py       # Synthetic ultrasound generator
│   │   ├── 📄 augmentation.py    # MONAI transforms
│   │   └── 📄 dataset.py         # PyTorch datasets
│   │
│   ├── 📁 models/                # AI Models
│   │   ├── 📄 backbone.py        # Swin Transformer
│   │   ├── 📄 segmentation.py    # Endometrial segmentation
│   │   ├── 📄 vascularity.py     # Blood flow classifier
│   │   ├── 📄 fibrosis.py        # Scar pattern detector
│   │   ├── 📄 temporal.py        # Video frame aggregation
│   │   └── 📄 unified.py         # Combined model
│   │
│   ├── 📁 agents/                # Clinical Decision
│   │   ├── 📄 decision.py        # Main decision agent
│   │   ├── 📄 risk_scorer.py     # Risk calculation
│   │   └── 📄 candidacy.py       # UG-IHI candidacy
│   │
│   ├── 📁 explainability/        # Model Explainability
│   │   ├── 📄 gradcam.py         # GradCAM++ implementation
│   │   └── 📄 attention.py       # Attention visualization
│   │
│   ├── 📁 reports/               # Clinical Reports
│   │   └── 📄 generator.py       # HTML/PDF generation
│   │
│   └── 📁 sdk/                   # Python SDK
│       └── 📄 client.py          # API client
│
└── 📁 tests/                      # Test Suite
    ├── 📄 test_preprocessing.py  # Data pipeline tests
    ├── 📄 test_models.py         # Model architecture tests
    ├── 📄 test_agents.py         # Clinical agent tests
    └── 📄 test_api.py            # API endpoint tests
```

---

## 📊 Clinical Thresholds

### Endometrial Thickness

```
                    Thickness Scale (mm)
    ├─────────┬─────────────────────────┬────────────┤
    0         5                        10           15+
    │  THIN   │        NORMAL          │   THICK    │
    │ ⚠️ Alert│          ✅             │  ⚠️ Alert  │
    └─────────┴─────────────────────────┴────────────┘
```

### Vascularity Types

| Type | Name | Description | Risk Level |
|------|------|-------------|------------|
| **0** | Avascular | No detectable blood flow | ⚠️ Attention |
| **I** | Minimal | Sparse vessels, low flow | ✅ Normal |
| **II** | Moderate | Moderate vasculature | ✅ Normal |
| **III** | High | Dense vessels, high flow | ⚠️ Alert |

### Candidacy Status

| Status | Risk Score | Recommendation |
|--------|------------|----------------|
| 🟢 **Excellent Candidate** | < 0.15 | Proceed with standard protocol |
| 🟡 **Good Candidate** | 0.15 - 0.30 | Proceed with monitoring |
| 🟠 **Cautionary** | 0.30 - 0.50 | Additional evaluation recommended |
| 🔴 **Not Recommended** | > 0.50 | Consider alternative treatments |

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# ═══════════════════════════════════════════════════════════
#                    MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════
MODEL_BACKBONE=swin_tiny          # swin_tiny, swin_small, swin_base
PRETRAINED_WEIGHTS=imagenet       # imagenet, none, /path/to/checkpoint
DEVICE=cuda                       # cuda, cpu

# ═══════════════════════════════════════════════════════════
#                   PROCESSING SETTINGS
# ═══════════════════════════════════════════════════════════
INPUT_SIZE=224                    # Input image size (square)
VIDEO_FPS=5                       # Frames per second for video
MAX_VIDEO_FRAMES=100              # Maximum frames to process

# ═══════════════════════════════════════════════════════════
#                      API SETTINGS
# ═══════════════════════════════════════════════════════════
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["*"]

# ═══════════════════════════════════════════════════════════
#                    REPORT SETTINGS
# ═══════════════════════════════════════════════════════════
REPORT_OUTPUT_DIR=./reports
REPORT_FORMAT=html                # html, pdf, both

# ═══════════════════════════════════════════════════════════
#                  CLINICAL THRESHOLDS
# ═══════════════════════════════════════════════════════════
THICKNESS_MIN_NORMAL=5.0
THICKNESS_MAX_NORMAL=10.0
THICKNESS_ALERT_HIGH=12.0
VASCULARITY_ALERT_THRESHOLD=3
FIBROSIS_ALERT_THRESHOLD=0.5
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src/uterus_scope --cov-report=html

# Run specific test modules
pytest tests/test_models.py -v
pytest tests/test_agents.py -v

# Run tests matching pattern
pytest tests/ -k "vascularity" -v
```

### Test Coverage Goals

| Module | Target Coverage |
|--------|----------------|
| `data/` | > 80% |
| `models/` | > 75% |
| `agents/` | > 85% |
| `api/` | > 70% |

---

## 📄 License & Disclaimer

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Medical Disclaimer

> ⚠️ **IMPORTANT**: This software is provided for **research and educational purposes only**.
>
> - It has **NOT** been cleared or approved by any regulatory authority (FDA, CE, etc.) for clinical use
> - It is **NOT** intended to diagnose, treat, cure, or prevent any disease
> - All outputs **MUST** be reviewed by qualified healthcare professionals
> - Clinical decisions should **NEVER** be based solely on this software's outputs
> - The developers assume **NO** liability for clinical use of this software

---

## 🙏 Acknowledgments

| Library | Purpose |
|---------|---------|
| [MONAI](https://monai.io/) | Medical imaging transforms |
| [PyTorch](https://pytorch.org/) | Deep learning framework |
| [timm](https://github.com/huggingface/pytorch-image-models) | Vision Transformer models |
| [FastAPI](https://fastapi.tiangolo.com/) | REST API framework |
| [WeasyPrint](https://weasyprint.org/) | PDF generation |

---

<div align="center">

**Made with ❤️ for advancing women's health**

[Report Bug](https://github.com/uterus-scope-ai/uterus-scope-ai/issues) · [Request Feature](https://github.com/uterus-scope-ai/uterus-scope-ai/issues) · [Documentation](https://uterus-scope-ai.readthedocs.io)

</div>

"""
UterusScope-AI: Autonomous Ultrasound Interpretation Agent

An open-source medical AI system for analyzing transvaginal ultrasound
to assess endometrial thickness, vascularity, and fibrosis risk.
"""

__version__ = "0.1.0"
__author__ = "UterusScope-AI Team"

from uterus_scope.config import Config, get_config
from uterus_scope.sdk.client import UterusScopeClient

# Lazy imports for heavy modules
def get_model():
    """Get the unified UterusScope model."""
    from uterus_scope.models.unified import UterusScopeModel
    return UterusScopeModel

def get_decision_agent():
    """Get the clinical decision agent."""
    from uterus_scope.agents.decision import ClinicalDecisionAgent
    return ClinicalDecisionAgent

__all__ = [
    "UterusScopeClient",
    "Config",
    "get_config",
    "get_model",
    "get_decision_agent",
    "__version__",
]

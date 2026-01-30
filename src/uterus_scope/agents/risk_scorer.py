"""
Risk scoring module for clinical assessments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from uterus_scope.config import get_config


class RiskCategory(str, Enum):
    """Risk categories."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class RiskFactor:
    """Individual risk factor."""
    name: str
    score: float
    weight: float
    description: str = ""


@dataclass
class RiskAssessment:
    """Complete risk assessment."""
    total_score: float
    category: RiskCategory
    factors: list[RiskFactor] = field(default_factory=list)
    confidence: float = 0.0
    recommendations: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'total_score': self.total_score,
            'category': self.category.value,
            'factors': [{'name': f.name, 'score': f.score, 'weight': f.weight} for f in self.factors],
            'confidence': self.confidence,
            'recommendations': self.recommendations,
        }


class RiskScorer:
    """Risk scoring calculator."""
    
    DEFAULT_WEIGHTS = {'thickness': 0.25, 'vascularity': 0.25, 'fibrosis': 0.30, 'image_quality': 0.10, 'patient_factors': 0.10}
    THRESHOLDS = {RiskCategory.LOW: 0.2, RiskCategory.MODERATE: 0.4, RiskCategory.HIGH: 0.6}
    
    def __init__(self, weights: Optional[dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.config = get_config()
    
    def calculate_risk(self, thickness_mm: float, vascularity_type: int, fibrosis_score: float, image_quality: float = 1.0) -> RiskAssessment:
        factors = []
        t = self.config.clinical
        
        # Thickness risk
        if thickness_mm < t.thickness_min_normal:
            thickness_risk = min(1.0, (t.thickness_min_normal - thickness_mm) / 3)
        elif thickness_mm > t.thickness_alert_high:
            thickness_risk = 0.8
        elif thickness_mm > t.thickness_max_normal:
            thickness_risk = 0.4
        else:
            thickness_risk = 0.0
        factors.append(RiskFactor("Thickness", thickness_risk, self.weights['thickness']))
        
        # Vascularity risk
        vasc_risk = min(1.0, vascularity_type * 0.3)
        factors.append(RiskFactor("Vascularity", vasc_risk, self.weights['vascularity']))
        
        # Fibrosis risk
        factors.append(RiskFactor("Fibrosis", min(1.0, fibrosis_score * 1.5), self.weights['fibrosis']))
        
        # Image quality
        factors.append(RiskFactor("Image Quality", 1.0 - image_quality, self.weights['image_quality']))
        
        # Calculate total
        total_score = sum(f.score * f.weight for f in factors) / sum(f.weight for f in factors)
        
        # Categorize
        if total_score < 0.2:
            category = RiskCategory.LOW
        elif total_score < 0.4:
            category = RiskCategory.MODERATE
        elif total_score < 0.6:
            category = RiskCategory.HIGH
        else:
            category = RiskCategory.VERY_HIGH
        
        recommendations = self._get_recommendations(category)
        
        return RiskAssessment(total_score=total_score, category=category, factors=factors, 
                             confidence=image_quality * 0.9, recommendations=recommendations)
    
    def _get_recommendations(self, category: RiskCategory) -> list[str]:
        if category == RiskCategory.LOW:
            return ["Proceed with standard protocol"]
        elif category == RiskCategory.MODERATE:
            return ["Consider additional evaluation", "Proceed with monitoring"]
        elif category == RiskCategory.HIGH:
            return ["Recommend specialist consultation"]
        else:
            return ["UG-IHI may not be appropriate", "Refer for full evaluation"]

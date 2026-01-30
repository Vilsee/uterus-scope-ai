"""
UG-IHI candidacy assessment module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from uterus_scope.config import CandidacyStatus, get_config
from uterus_scope.agents.risk_scorer import RiskScorer, RiskCategory


@dataclass
class CandidacyResult:
    """Candidacy assessment result."""
    status: CandidacyStatus
    confidence: float
    summary: str
    contraindications: list[str]
    recommendations: list[str]
    
    def to_dict(self) -> dict:
        return {
            'status': self.status.value,
            'confidence': self.confidence,
            'summary': self.summary,
            'contraindications': self.contraindications,
            'recommendations': self.recommendations,
        }


class CandidacyAssessor:
    """
    UG-IHI therapy candidacy assessor.
    
    Evaluates patient suitability based on ultrasound findings.
    """
    
    def __init__(self):
        self.config = get_config()
        self.risk_scorer = RiskScorer()
    
    def assess(
        self,
        thickness_mm: float,
        vascularity_type: int,
        fibrosis_score: float,
        model_confidence: float = 1.0,
    ) -> CandidacyResult:
        """
        Assess candidacy for UG-IHI therapy.
        
        Args:
            thickness_mm: Endometrial thickness
            vascularity_type: Vascularity classification (0-3)
            fibrosis_score: Fibrosis severity (0-1)
            model_confidence: Model prediction confidence
            
        Returns:
            CandidacyResult with status and recommendations
        """
        contraindications = []
        recommendations = []
        
        # Check contraindications
        t = self.config.clinical
        
        if thickness_mm > t.thickness_alert_high:
            contraindications.append(f"Endometrial thickness ({thickness_mm:.1f}mm) exceeds safe threshold")
        
        if vascularity_type >= t.vascularity_alert_threshold:
            contraindications.append("High vascularity increases bleeding risk")
        
        if fibrosis_score > t.fibrosis_alert_threshold:
            contraindications.append("Significant fibrosis may affect treatment efficacy")
        
        # Calculate risk
        risk = self.risk_scorer.calculate_risk(thickness_mm, vascularity_type, fibrosis_score, model_confidence)
        
        # Determine status
        if len(contraindications) >= 2 or risk.category == RiskCategory.VERY_HIGH:
            status = CandidacyStatus.NOT_RECOMMENDED
            summary = "UG-IHI therapy is not recommended due to multiple risk factors."
        elif len(contraindications) == 1 or risk.category == RiskCategory.HIGH:
            status = CandidacyStatus.CAUTIONARY
            summary = "Proceed with caution. Additional evaluation recommended."
        elif risk.category == RiskCategory.MODERATE:
            status = CandidacyStatus.GOOD_CANDIDATE
            summary = "Patient is a good candidate with minor considerations."
        else:
            status = CandidacyStatus.EXCELLENT_CANDIDATE
            summary = "Patient is an excellent candidate for UG-IHI therapy."
        
        # Generate recommendations
        if status == CandidacyStatus.EXCELLENT_CANDIDATE:
            recommendations.append("Proceed with standard UG-IHI protocol")
        elif status == CandidacyStatus.GOOD_CANDIDATE:
            recommendations.append("Proceed with close monitoring")
            if thickness_mm < t.thickness_min_normal:
                recommendations.append("Consider hormonal preparation")
        elif status == CandidacyStatus.CAUTIONARY:
            recommendations.append("Consult with specialist before proceeding")
            recommendations.append("Consider alternative treatments")
        else:
            recommendations.append("Refer for comprehensive gynecological evaluation")
            recommendations.append("Explore alternative treatment options")
        
        confidence = model_confidence * (1.0 - risk.total_score * 0.2)
        
        return CandidacyResult(
            status=status,
            confidence=confidence,
            summary=summary,
            contraindications=contraindications,
            recommendations=recommendations,
        )

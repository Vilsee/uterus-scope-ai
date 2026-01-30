"""
Clinical decision agent for UterusScope-AI.

Aggregates model outputs into clinical recommendations
for UG-IHI therapy candidacy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from uterus_scope.config import get_config, ClinicalThresholds, CandidacyStatus
from uterus_scope.models.unified import ModelOutput


class AlertLevel(str, Enum):
    """Clinical alert levels."""
    NORMAL = "normal"
    ATTENTION = "attention"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Finding:
    """Individual clinical finding."""
    name: str
    value: float | str
    unit: Optional[str] = None
    alert_level: AlertLevel = AlertLevel.NORMAL
    description: str = ""
    recommendation: str = ""


@dataclass
class DecisionResult:
    """Result from clinical decision agent."""
    
    # Overall candidacy recommendation
    candidacy: CandidacyStatus
    candidacy_confidence: float
    
    # Individual findings
    findings: list[Finding] = field(default_factory=list)
    
    # Alert flags
    alerts: list[str] = field(default_factory=list)
    
    # Summary text for clinicians
    summary: str = ""
    
    # Detailed reasoning
    reasoning: list[str] = field(default_factory=list)
    
    # Contraindications found
    contraindications: list[str] = field(default_factory=list)
    
    # Overall risk score 0-1
    risk_score: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            'candidacy': self.candidacy.value,
            'candidacy_confidence': self.candidacy_confidence,
            'findings': [
                {
                    'name': f.name,
                    'value': f.value,
                    'unit': f.unit,
                    'alert_level': f.alert_level.value,
                    'description': f.description,
                    'recommendation': f.recommendation,
                }
                for f in self.findings
            ],
            'alerts': self.alerts,
            'summary': self.summary,
            'reasoning': self.reasoning,
            'contraindications': self.contraindications,
            'risk_score': self.risk_score,
        }


class ClinicalDecisionAgent:
    """
    Autonomous clinical decision agent.
    
    Aggregates model outputs into clinical recommendations
    by evaluating measurements against clinical thresholds
    and applying decision logic.
    
    Attributes:
        thresholds: Clinical threshold configuration
    """
    
    def __init__(
        self,
        thresholds: Optional[ClinicalThresholds] = None,
    ):
        """
        Initialize the decision agent.
        
        Args:
            thresholds: Clinical thresholds (uses config defaults if None)
        """
        config = get_config()
        self.thresholds = thresholds or config.clinical
    
    def evaluate(
        self,
        model_output: ModelOutput,
        patient_history: Optional[dict] = None,
    ) -> DecisionResult:
        """
        Evaluate model output and generate clinical decision.
        
        Args:
            model_output: Complete model output
            patient_history: Optional patient history data
            
        Returns:
            DecisionResult with recommendations
        """
        findings = []
        alerts = []
        contraindications = []
        reasoning = []
        
        # Extract values (handle batched output)
        thickness = self._get_scalar(model_output.segmentation.thickness_mm)
        vasc_type = self._get_scalar(model_output.vascularity.predicted_type)
        vasc_conf = self._get_scalar(model_output.vascularity.confidence)
        fibrosis = self._get_scalar(model_output.fibrosis.severity_score)
        has_fibrosis = self._get_scalar(model_output.fibrosis.has_fibrosis)
        
        # Evaluate endometrial thickness
        thickness_finding, thickness_alert = self._evaluate_thickness(thickness)
        findings.append(thickness_finding)
        if thickness_alert:
            alerts.append(thickness_alert)
            reasoning.append(f"Thickness of {thickness:.1f}mm is outside normal range (5-10mm)")
        
        # Evaluate vascularity
        vasc_finding, vasc_alert = self._evaluate_vascularity(vasc_type, vasc_conf)
        findings.append(vasc_finding)
        if vasc_alert:
            alerts.append(vasc_alert)
            if vasc_type >= 3:
                contraindications.append("High vascularity may increase bleeding risk")
                reasoning.append("Type III vascularity indicates high blood flow")
        
        # Evaluate fibrosis
        fibrosis_finding, fibrosis_alert = self._evaluate_fibrosis(fibrosis, has_fibrosis)
        findings.append(fibrosis_finding)
        if fibrosis_alert:
            alerts.append(fibrosis_alert)
            reasoning.append(f"Fibrosis score {fibrosis:.2f} indicates significant scarring")
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(
            thickness, vasc_type, fibrosis,
        )
        
        # Determine candidacy
        candidacy, confidence = self._determine_candidacy(
            thickness, vasc_type, fibrosis,
            contraindications, risk_score,
        )
        
        # Generate summary
        summary = self._generate_summary(
            candidacy, findings, contraindications,
        )
        
        return DecisionResult(
            candidacy=candidacy,
            candidacy_confidence=confidence,
            findings=findings,
            alerts=alerts,
            summary=summary,
            reasoning=reasoning,
            contraindications=contraindications,
            risk_score=risk_score,
        )
    
    def _get_scalar(self, tensor_or_value) -> float:
        """Extract scalar value from tensor or return as-is."""
        if hasattr(tensor_or_value, 'item'):
            return tensor_or_value.item()
        if hasattr(tensor_or_value, '__getitem__'):
            return float(tensor_or_value[0])
        return float(tensor_or_value) if tensor_or_value is not None else 0.0
    
    def _evaluate_thickness(
        self,
        thickness: float,
    ) -> tuple[Finding, Optional[str]]:
        """Evaluate endometrial thickness."""
        alert = None
        alert_level = AlertLevel.NORMAL
        description = "Within normal range"
        recommendation = ""
        
        if thickness < self.thresholds.thickness_min_normal:
            alert_level = AlertLevel.WARNING
            description = "Below normal range - thin endometrium"
            recommendation = "Consider hormonal preparation before procedure"
            alert = f"Low endometrial thickness: {thickness:.1f}mm"
        elif thickness > self.thresholds.thickness_alert_high:
            alert_level = AlertLevel.CRITICAL
            description = "Significantly above normal - investigate cause"
            recommendation = "Rule out pathology before proceeding"
            alert = f"High endometrial thickness: {thickness:.1f}mm"
        elif thickness > self.thresholds.thickness_max_normal:
            alert_level = AlertLevel.ATTENTION
            description = "Above normal range"
            recommendation = "Monitor and consider evaluation"
        
        return Finding(
            name="Endometrial Thickness",
            value=round(thickness, 1),
            unit="mm",
            alert_level=alert_level,
            description=description,
            recommendation=recommendation,
        ), alert
    
    def _evaluate_vascularity(
        self,
        vasc_type: int,
        confidence: float,
    ) -> tuple[Finding, Optional[str]]:
        """Evaluate vascularity pattern."""
        alert = None
        
        type_names = {
            0: "Type 0 (Avascular)",
            1: "Type I (Minimal)",
            2: "Type II (Moderate)",
            3: "Type III (High)",
        }
        
        type_name = type_names.get(int(vasc_type), f"Type {vasc_type}")
        
        if vasc_type >= self.thresholds.vascularity_alert_threshold:
            alert_level = AlertLevel.WARNING
            description = "High vascularity detected"
            recommendation = "Increased bleeding risk during procedure"
            alert = f"High vascularity: {type_name}"
        elif vasc_type == 0:
            alert_level = AlertLevel.ATTENTION
            description = "No detectable blood flow"
            recommendation = "Verify image quality and probe positioning"
        else:
            alert_level = AlertLevel.NORMAL
            description = "Normal vascularity pattern"
            recommendation = ""
        
        return Finding(
            name="Vascularity",
            value=type_name,
            unit=None,
            alert_level=alert_level,
            description=f"{description} (confidence: {confidence:.0%})",
            recommendation=recommendation,
        ), alert
    
    def _evaluate_fibrosis(
        self,
        score: float,
        has_fibrosis: Optional[float],
    ) -> tuple[Finding, Optional[str]]:
        """Evaluate fibrosis/scarring."""
        alert = None
        
        if score > self.thresholds.fibrosis_alert_threshold:
            alert_level = AlertLevel.WARNING
            description = "Significant fibrosis detected"
            recommendation = "May affect hydrogel distribution"
            alert = f"High fibrosis score: {score:.2f}"
        elif score > self.thresholds.fibrosis_normal_max:
            alert_level = AlertLevel.ATTENTION
            description = "Mild to moderate fibrosis"
            recommendation = "Monitor for treatment response"
        else:
            alert_level = AlertLevel.NORMAL
            description = "No significant fibrosis"
            recommendation = ""
        
        severity = "None/Minimal"
        if score >= 0.8:
            severity = "Severe"
        elif score >= 0.6:
            severity = "Significant"
        elif score >= 0.4:
            severity = "Moderate"
        elif score >= 0.2:
            severity = "Mild"
        
        return Finding(
            name="Fibrosis",
            value=f"{severity} ({score:.2f})",
            unit=None,
            alert_level=alert_level,
            description=description,
            recommendation=recommendation,
        ), alert
    
    def _calculate_risk_score(
        self,
        thickness: float,
        vasc_type: int,
        fibrosis: float,
    ) -> float:
        """Calculate overall risk score 0-1."""
        risk = 0.0
        
        # Thickness contribution
        if thickness < self.thresholds.thickness_min_normal:
            risk += 0.2 * (self.thresholds.thickness_min_normal - thickness) / 3
        elif thickness > self.thresholds.thickness_alert_high:
            risk += 0.3
        elif thickness > self.thresholds.thickness_max_normal:
            risk += 0.1
        
        # Vascularity contribution
        risk += 0.2 * (vasc_type / 3)
        
        # Fibrosis contribution
        risk += 0.3 * fibrosis
        
        return min(1.0, max(0.0, risk))
    
    def _determine_candidacy(
        self,
        thickness: float,
        vasc_type: int,
        fibrosis: float,
        contraindications: list[str],
        risk_score: float,
    ) -> tuple[CandidacyStatus, float]:
        """Determine UG-IHI candidacy status."""
        
        # Critical contraindications
        if len(contraindications) > 0:
            if risk_score > 0.6:
                return CandidacyStatus.NOT_RECOMMENDED, 0.8
            return CandidacyStatus.CAUTIONARY, 0.7
        
        # Score-based classification
        if risk_score < 0.15:
            return CandidacyStatus.EXCELLENT_CANDIDATE, 0.9
        elif risk_score < 0.3:
            return CandidacyStatus.GOOD_CANDIDATE, 0.8
        elif risk_score < 0.5:
            return CandidacyStatus.CAUTIONARY, 0.7
        else:
            return CandidacyStatus.NOT_RECOMMENDED, 0.75
    
    def _generate_summary(
        self,
        candidacy: CandidacyStatus,
        findings: list[Finding],
        contraindications: list[str],
    ) -> str:
        """Generate clinician-readable summary."""
        
        candidacy_text = {
            CandidacyStatus.EXCELLENT_CANDIDATE: 
                "Patient appears to be an excellent candidate for UG-IHI therapy.",
            CandidacyStatus.GOOD_CANDIDATE:
                "Patient appears to be a good candidate for UG-IHI therapy with minor considerations.",
            CandidacyStatus.CAUTIONARY:
                "Caution advised. Additional evaluation recommended before proceeding with UG-IHI therapy.",
            CandidacyStatus.NOT_RECOMMENDED:
                "UG-IHI therapy is not recommended based on current assessment.",
        }
        
        summary_parts = [candidacy_text[candidacy]]
        
        # Add key findings
        abnormal = [f for f in findings if f.alert_level != AlertLevel.NORMAL]
        if abnormal:
            finding_text = ", ".join([f.name for f in abnormal])
            summary_parts.append(f"Notable findings: {finding_text}.")
        
        # Add contraindications
        if contraindications:
            summary_parts.append(f"Considerations: {'; '.join(contraindications)}.")
        
        return " ".join(summary_parts)

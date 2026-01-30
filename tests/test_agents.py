"""Tests for clinical decision agents."""

import pytest
import torch

from uterus_scope.agents.decision import ClinicalDecisionAgent, AlertLevel
from uterus_scope.agents.risk_scorer import RiskScorer, RiskCategory
from uterus_scope.agents.candidacy import CandidacyAssessor
from uterus_scope.config import CandidacyStatus
from uterus_scope.models.unified import ModelOutput
from uterus_scope.models.segmentation import SegmentationOutput
from uterus_scope.models.vascularity import VascularityOutput
from uterus_scope.models.fibrosis import FibrosisOutput


def create_mock_output(thickness=8.0, vasc_type=1, fibrosis=0.2, confidence=0.9):
    """Create mock ModelOutput for testing."""
    return ModelOutput(
        segmentation=SegmentationOutput(
            mask=torch.randn(1, 1, 224, 224),
            thickness_mm=torch.tensor([thickness]),
        ),
        vascularity=VascularityOutput(
            probabilities=torch.tensor([[0.1, 0.6, 0.2, 0.1]]),
            predicted_type=torch.tensor([vasc_type]),
            confidence=torch.tensor([confidence]),
        ),
        fibrosis=FibrosisOutput(
            probability_map=torch.randn(1, 1, 224, 224),
            severity_score=torch.tensor([fibrosis]),
            has_fibrosis=torch.tensor([fibrosis > 0.3]),
        ),
        overall_confidence=torch.tensor([confidence]),
    )


class TestClinicalDecisionAgent:
    """Tests for clinical decision agent."""
    
    @pytest.fixture
    def agent(self):
        return ClinicalDecisionAgent()
    
    def test_normal_case(self, agent):
        output = create_mock_output(thickness=8.0, vasc_type=1, fibrosis=0.1)
        result = agent.evaluate(output)
        
        assert result.candidacy in [CandidacyStatus.EXCELLENT_CANDIDATE, CandidacyStatus.GOOD_CANDIDATE]
        assert len(result.contraindications) == 0
        assert result.risk_score < 0.3
    
    def test_high_vascularity(self, agent):
        output = create_mock_output(thickness=8.0, vasc_type=3, fibrosis=0.1)
        result = agent.evaluate(output)
        
        assert len(result.alerts) > 0
        assert any("vascularity" in a.lower() for a in result.alerts)
    
    def test_thick_endometrium(self, agent):
        output = create_mock_output(thickness=18.0, vasc_type=1, fibrosis=0.1)
        result = agent.evaluate(output)
        
        assert len(result.alerts) > 0
        assert result.candidacy in [CandidacyStatus.CAUTIONARY, CandidacyStatus.NOT_RECOMMENDED]
    
    def test_high_fibrosis(self, agent):
        output = create_mock_output(thickness=8.0, vasc_type=1, fibrosis=0.7)
        result = agent.evaluate(output)
        
        assert len(result.alerts) > 0


class TestRiskScorer:
    """Tests for risk scorer."""
    
    @pytest.fixture
    def scorer(self):
        return RiskScorer()
    
    def test_low_risk(self, scorer):
        result = scorer.calculate_risk(
            thickness_mm=7.0,
            vascularity_type=1,
            fibrosis_score=0.1,
        )
        assert result.category == RiskCategory.LOW
        assert result.total_score < 0.2
    
    def test_high_risk(self, scorer):
        result = scorer.calculate_risk(
            thickness_mm=16.0,
            vascularity_type=3,
            fibrosis_score=0.7,
        )
        assert result.category in [RiskCategory.HIGH, RiskCategory.VERY_HIGH]
        assert result.total_score > 0.4


class TestCandidacyAssessor:
    """Tests for candidacy assessor."""
    
    @pytest.fixture
    def assessor(self):
        return CandidacyAssessor()
    
    def test_excellent_candidate(self, assessor):
        result = assessor.assess(
            thickness_mm=7.5,
            vascularity_type=1,
            fibrosis_score=0.1,
        )
        assert result.status == CandidacyStatus.EXCELLENT_CANDIDATE
        assert len(result.contraindications) == 0
    
    def test_not_recommended(self, assessor):
        result = assessor.assess(
            thickness_mm=20.0,
            vascularity_type=3,
            fibrosis_score=0.8,
        )
        assert result.status == CandidacyStatus.NOT_RECOMMENDED
        assert len(result.contraindications) >= 2

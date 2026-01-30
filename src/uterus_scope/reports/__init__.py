"""Reports module for UterusScope-AI."""

from uterus_scope.reports.generator import (
    ClinicalReportGenerator,
    generate_report,
)

__all__ = [
    "ClinicalReportGenerator",
    "generate_report",
]

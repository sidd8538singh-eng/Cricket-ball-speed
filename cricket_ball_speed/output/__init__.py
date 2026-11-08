"""
Output Generation Subsystem for Cricket Ball Speed Measurement.
Generates:
  1. Final processed broadcast video (speed & date/time only, upper-left badge, freeze frame).
  2. Diagnostic video (raw/accepted/rejected balls, observed trajectory, predicted marks, 3D fit, HUD).
  3. JSON report (complete serialization of calibration, detections, timestamps, release, 3D points, speeds, QC).
  4. Excel benchmark report (styled delivery summary, radar errors, clearly defined variance, quality metrics).
"""

from .final_video import FinalVideoRenderer, generate_final_video
from .diagnostic_video import DiagnosticVideoRenderer, generate_diagnostic_video
from .json_report import generate_json_report
from .excel_report import ExcelReportWriter, EXCEL_HEADERS

__all__ = [
    "FinalVideoRenderer",
    "generate_final_video",
    "DiagnosticVideoRenderer",
    "generate_diagnostic_video",
    "generate_json_report",
    "ExcelReportWriter",
    "EXCEL_HEADERS",
]

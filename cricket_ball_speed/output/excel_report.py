"""
Excel Report Generator.
Produces formatted Excel (.xlsx) workbooks summarizing bowling delivery measurements,
radar gun benchmarks, calibration accuracy, tracking counts, and quality control audit reasons.

INCLUDES STRICT COLUMNS:
1. Video Filename
2. Filename-derived Input Reference Speed (if configured)
3. Radar Speed
4. Calculated Release Speed
5. C1-C2 Average Speed
6. Difference (Calculated - Radar)
7. Absolute Error |Calculated - Radar|
8. Variance (Residual Error^2 in km^2/h^2, with clearly defined mathematical meaning)
9. Calibration Status
10. Reprojection Error (px)
11. Detected Frame Count
12. Predicted Frame Count
13. Release Confidence
14. Trajectory Quality (RMSE px)
15. Speed Uncertainty (+-km/h)
16. Measurement Status (MEASURED / NOT_MEASURED)
17. Rejection Reason

DECOUPLING PRINCIPLE:
Independent of detection, tracking, and optimization engines. Accepts plain dictionaries or objects.
"""

from typing import List, Dict, Any, Optional
import os
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


EXCEL_HEADERS = [
    "Video Filename",
    "Reference Speed (km/h)",
    "Radar Speed (km/h)",
    "Calculated Release Speed (km/h)",
    "C1-C2 Avg Speed (km/h)",
    "Difference (km/h)",
    "Absolute Error (km/h)",
    "Variance (Error^2: km^2/h^2)*",
    "Calibration Status",
    "Reprojection Error (px)",
    "Detected Frame Count",
    "Predicted Frame Count",
    "Release Confidence",
    "Trajectory Quality (RMSE px)",
    "Speed Uncertainty (±km/h)",
    "Measurement Status",
    "Rejection Reason",
]


class ExcelReportWriter:
    """Writes and appends delivery results to a styled Excel workbook."""

    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        os.makedirs(os.path.dirname(os.path.abspath(excel_path)), exist_ok=True)
        self.wb = openpyxl.Workbook()
        self.ws = self.wb.active
        self.ws.title = "Delivery Speed Audit"
        self._setup_headers()

    def _setup_headers(self):
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
        header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

        self.ws.row_dimensions[1].height = 28
        for col_idx, header in enumerate(EXCEL_HEADERS, start=1):
            cell = self.ws.cell(row=1, column=col_idx, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_align

    def add_delivery_row(self, row_data: Dict[str, Any]):
        """
        Appends a delivery evaluation record to the workbook.
        """
        row_idx = self.ws.max_row + 1
        self.ws.row_dimensions[row_idx].height = 20

        # Extract values (supports diverse key naming for decoupling)
        filename = row_data.get("filename") or row_data.get("video_name") or ""
        ref_speed = row_data.get("reference_speed") or row_data.get("filename_reference_speed")
        radar_speed = row_data.get("radar_speed") or row_data.get("radar_speed_kmh")
        calc_speed = row_data.get("calculated_speed") or row_data.get("release_speed_kmh") or row_data.get("speed_km_h")
        c1c2_speed = row_data.get("c1_c2_speed") or row_data.get("c1_c2_speed_kmh") or row_data.get("average_speed_km_h")

        final_status = row_data.get("final_status") or row_data.get("status") or ("MEASURED" if calc_speed is not None else "NOT_MEASURED")
        is_measured = (final_status == "MEASURED")

        # Compute error metrics against radar ground truth if available
        diff = None
        abs_err = None
        variance = None

        benchmark_target = radar_speed if radar_speed is not None else ref_speed

        if is_measured and calc_speed is not None and benchmark_target is not None:
            diff = round(float(calc_speed) - float(benchmark_target), 2)
            abs_err = round(abs(diff), 2)
            # Clearly defined meaning: Variance is the squared residual error (e_i^2) in km^2/h^2
            variance = round(diff ** 2, 3)

        calib_status = row_data.get("calibration_status") or "VALID"
        reproj_err = row_data.get("reprojection_error_px") or row_data.get("reprojection_error")
        det_frames = row_data.get("detected_frames") or row_data.get("detected_frame_count") or 0
        pred_frames = row_data.get("predicted_frames") or row_data.get("predicted_frame_count") or 0
        rel_conf = row_data.get("release_confidence")
        traj_rmse = row_data.get("trajectory_rmse_px") or row_data.get("trajectory_quality")
        speed_unc = row_data.get("speed_uncertainty_kmh") or row_data.get("speed_uncertainty")
        rejection_reason = row_data.get("rejection_reason") or ("None (Passed QC)" if is_measured else "REJECTED")

        values = [
            filename,
            round(float(ref_speed), 2) if ref_speed is not None else "N/A",
            round(float(radar_speed), 2) if radar_speed is not None else "N/A",
            round(float(calc_speed), 2) if (calc_speed is not None and is_measured) else "NOT_MEASURED",
            round(float(c1c2_speed), 2) if c1c2_speed is not None else "N/A",
            diff if diff is not None else "N/A",
            abs_err if abs_err is not None else "N/A",
            variance if variance is not None else "N/A",
            calib_status,
            round(float(reproj_err), 2) if reproj_err is not None else "N/A",
            det_frames,
            pred_frames,
            round(float(rel_conf), 2) if rel_conf is not None else "N/A",
            round(float(traj_rmse), 2) if traj_rmse is not None else "N/A",
            round(float(speed_unc), 2) if speed_unc is not None else "N/A",
            final_status,
            rejection_reason,
        ]

        # Alternating zebra striping
        zebra_fill = PatternFill(start_color="F9FBFD", end_color="F9FBFD", fill_type="solid") if (row_idx % 2 == 0) else None

        for col_idx, val in enumerate(values, start=1):
            cell = self.ws.cell(row=row_idx, column=col_idx, value=val)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if zebra_fill:
                cell.fill = zebra_fill

            # Highlight Measurement Status Column
            if col_idx == 16:  # Measurement Status
                if val == "MEASURED":
                    cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                    cell.font = Font(name="Calibri", color="006100", bold=True)
                else:
                    cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
                    cell.font = Font(name="Calibri", color="9C0006", bold=True)

    def save(self) -> str:
        """Adds footnote definition and auto-adjusts column widths before saving."""
        last_row = self.ws.max_row

        # Add footnote defining Variance
        note_row = last_row + 2
        note_cell = self.ws.cell(
            row=note_row, column=1,
            value="* Mathematical Definition of Variance: Squared residual benchmark error (Calculated Speed - Radar Speed)^2 in km^2/h^2."
        )
        note_cell.font = Font(name="Calibri", size=9, italic=True, color="555555")

        # Auto-adjust column widths
        for col in self.ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                # Do not let long footnote blow up column 1 width
                if cell.row == note_row:
                    continue
                val_str = str(cell.value or "")
                max_len = max(max_len, len(val_str))
            self.ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

        self.wb.save(self.excel_path)
        return self.excel_path

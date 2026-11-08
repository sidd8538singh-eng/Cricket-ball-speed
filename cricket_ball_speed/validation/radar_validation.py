"""
Radar Validation: Independent benchmark comparison against calibrated radar gun data.

CRITICAL SCIENTIFIC MANDATES:
1. Radar speeds are strictly an independent ground truth for post-hoc evaluation.
2. Radar speed is NEVER used as an input to the computer-vision speed calculations.
3. No hardcoded radar values or video-specific multipliers are EVER applied.
4. Transparently records unmeasured deliveries and computes both measurement coverage and error metrics.
5. Evaluates:
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Square Error)
   - Bias (Mean Signed Error)
   - Percentage within +-2 km/h
   - Percentage within +-5 km/h
   - Measurement Coverage % and NOT_MEASURED %
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os
import csv
import json
import numpy as np

from .benchmark_plots import BenchmarkPlotGenerator


@dataclass
class DeliveryRadarComparison:
    """Individual delivery comparison record between calculated vision speed and radar."""
    video_name: str
    radar_speed_kmh: float
    vision_speed_kmh: Optional[float]
    c1_c2_speed_kmh: Optional[float]
    is_measured: bool
    status: str                         # "MEASURED" or "NOT_MEASURED"
    rejection_reason: Optional[str] = None
    signed_error_kmh: Optional[float] = None      # v_calc - v_radar
    absolute_error_kmh: Optional[float] = None    # |v_calc - v_radar|
    within_2kmh: Optional[bool] = None
    within_5kmh: Optional[bool] = None

    # Backward compatibility property aliases
    @property
    def error_kmh(self) -> Optional[float]:
        return self.signed_error_kmh

    @property
    def abs_error_kmh(self) -> Optional[float]:
        return self.absolute_error_kmh

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_name": self.video_name,
            "status": self.status,
            "is_measured": self.is_measured,
            "rejection_reason": self.rejection_reason,
            "radar_speed_kmh": round(float(self.radar_speed_kmh), 2),
            "vision_speed_kmh": round(float(self.vision_speed_kmh), 2) if self.vision_speed_kmh is not None else None,
            "c1_c2_speed_kmh": round(float(self.c1_c2_speed_kmh), 2) if self.c1_c2_speed_kmh is not None else None,
            "signed_error_kmh": round(float(self.signed_error_kmh), 2) if self.signed_error_kmh is not None else None,
            "absolute_error_kmh": round(float(self.absolute_error_kmh), 2) if self.absolute_error_kmh is not None else None,
            "within_2kmh": self.within_2kmh,
            "within_5kmh": self.within_5kmh,
        }


@dataclass
class RadarBenchmarkSummary:
    """Statistical benchmark summary across a dataset of deliveries."""
    total_deliveries: int
    measured_deliveries: int
    not_measured_deliveries: int
    coverage_percentage: float          # Percentage measured
    not_measured_rate: float            # Percentage unmeasured
    mae_kmh: float                      # Mean Absolute Error
    rmse_kmh: float                     # Root Mean Square Error
    bias_kmh: float                     # Mean Signed Error (Systematic Drift)
    within_2kmh_percentage: float       # Primary standard
    within_5kmh_percentage: float       # Secondary standard
    max_error_kmh: float
    min_error_kmh: float
    std_error_kmh: float
    comparisons: List[DeliveryRadarComparison] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_deliveries": self.total_deliveries,
            "measured_deliveries": self.measured_deliveries,
            "not_measured_deliveries": self.not_measured_deliveries,
            "coverage_percentage": round(float(self.coverage_percentage), 1),
            "not_measured_rate": round(float(self.not_measured_rate), 1),
            "mae_kmh": round(float(self.mae_kmh), 2),
            "rmse_kmh": round(float(self.rmse_kmh), 2),
            "bias_kmh": round(float(self.bias_kmh), 2),
            "within_2kmh_percentage": round(float(self.within_2kmh_percentage), 1),
            "within_5kmh_percentage": round(float(self.within_5kmh_percentage), 1),
            "max_error_kmh": round(float(self.max_error_kmh), 2),
            "min_error_kmh": round(float(self.min_error_kmh), 2),
            "std_error_kmh": round(float(self.std_error_kmh), 2),
            "deliveries": [c.to_dict() for c in self.comparisons],
        }

    def to_markdown_table(self) -> str:
        """Formats the delivery comparisons and KPI summary into GitHub Markdown."""
        lines = []
        lines.append("## Radar Ground Truth Benchmark Report\n")
        lines.append("| Metric | Value |")
        lines.append("| :--- | :--- |")
        lines.append(f"| **Total Deliveries** | {self.total_deliveries} |")
        lines.append(f"| **Measured Deliveries** | {self.measured_deliveries} ({self.coverage_percentage:.1f}%) |")
        lines.append(f"| **NOT_MEASURED Rate** | {self.not_measured_deliveries} ({self.not_measured_rate:.1f}%) |")
        lines.append(f"| **Mean Absolute Error (MAE)** | **{self.mae_kmh:.2f} km/h** |")
        lines.append(f"| **Root Mean Sq Error (RMSE)** | **{self.rmse_kmh:.2f} km/h** |")
        lines.append(f"| **System Bias** | **{self.bias_kmh:+.2f} km/h** |")
        lines.append(f"| **Within ±2.0 km/h** | **{self.within_2kmh_percentage:.1f}%** |")
        lines.append(f"| **Within ±5.0 km/h** | **{self.within_5kmh_percentage:.1f}%** |")
        lines.append(f"| **Max Error** | {self.max_error_kmh:.2f} km/h |\n")

        lines.append("### Delivery Breakdown\n")
        lines.append("| Video File | Status | Radar Speed | Vision Speed | C1-C2 Speed | Signed Error | Abs Error | ±2 km/h | Rejection Reason |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

        for c in self.comparisons:
            radar_str = f"{c.radar_speed_kmh:.1f}"
            vis_str = f"{c.vision_speed_kmh:.1f}" if c.vision_speed_kmh is not None else "-"
            c1c2_str = f"{c.c1_c2_speed_kmh:.1f}" if c.c1_c2_speed_kmh is not None else "-"
            s_err_str = f"{c.signed_error_kmh:+.1f}" if c.signed_error_kmh is not None else "-"
            a_err_str = f"{c.absolute_error_kmh:.1f}" if c.absolute_error_kmh is not None else "-"
            w2_str = "PASS" if c.within_2kmh else ("FAIL" if c.within_2kmh is not None else "-")
            reason_str = c.rejection_reason or "-"
            lines.append(
                f"| `{c.video_name}` | **{c.status}** | {radar_str} km/h | {vis_str} km/h | {c1c2_str} km/h | "
                f"{s_err_str} km/h | {a_err_str} km/h | {w2_str} | {reason_str} |"
            )

        return "\n".join(lines)

    def to_csv(self, filepath: str) -> str:
        """Exports delivery comparison records to a standard CSV file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        fieldnames = [
            "video_name", "status", "is_measured", "radar_speed_kmh",
            "vision_speed_kmh", "c1_c2_speed_kmh", "signed_error_kmh",
            "absolute_error_kmh", "within_2kmh", "within_5kmh", "rejection_reason"
        ]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in self.comparisons:
                writer.writerow(c.to_dict())
        return filepath

    def save_json(self, filepath: str) -> str:
        """Exports full benchmark summary and records to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return filepath


class RadarValidator:
    """
    Evaluates system accuracy against independent Doppler radar measurements.
    """

    def __init__(self):
        self.comparisons: List[DeliveryRadarComparison] = []

    def add_comparison(
        self,
        video_name: str,
        vision_speed_kmh: Optional[float],
        radar_speed_kmh: float,
        is_measured: bool,
        rejection_reason: Optional[str] = None,
        c1_c2_speed_kmh: Optional[float] = None
    ) -> DeliveryRadarComparison:
        """
        Adds a single delivery evaluation row.
        """
        radar_val = float(radar_speed_kmh)
        status = "MEASURED" if (is_measured and vision_speed_kmh is not None) else "NOT_MEASURED"

        signed_err: Optional[float] = None
        abs_err: Optional[float] = None
        w2: Optional[bool] = None
        w5: Optional[bool] = None

        if status == "MEASURED" and vision_speed_kmh is not None:
            v_val = float(vision_speed_kmh)
            signed_err = v_val - radar_val
            abs_err = abs(signed_err)
            w2 = abs_err <= 2.0
            w5 = abs_err <= 5.0

        comp = DeliveryRadarComparison(
            video_name=video_name,
            radar_speed_kmh=radar_val,
            vision_speed_kmh=float(vision_speed_kmh) if vision_speed_kmh is not None else None,
            c1_c2_speed_kmh=float(c1_c2_speed_kmh) if c1_c2_speed_kmh is not None else None,
            is_measured=(status == "MEASURED"),
            status=status,
            rejection_reason=rejection_reason,
            signed_error_kmh=signed_err,
            absolute_error_kmh=abs_err,
            within_2kmh=w2,
            within_5kmh=w5
        )
        self.comparisons.append(comp)
        return comp

    def compute_summary(self) -> RadarBenchmarkSummary:
        """
        Computes comprehensive aggregate statistics across all added comparisons.
        """
        total = len(self.comparisons)
        if total == 0:
            return RadarBenchmarkSummary(
                total_deliveries=0,
                measured_deliveries=0,
                not_measured_deliveries=0,
                coverage_percentage=0.0,
                not_measured_rate=0.0,
                mae_kmh=0.0,
                rmse_kmh=0.0,
                bias_kmh=0.0,
                within_2kmh_percentage=0.0,
                within_5kmh_percentage=0.0,
                max_error_kmh=0.0,
                min_error_kmh=0.0,
                std_error_kmh=0.0,
                comparisons=[]
            )

        measured = [c for c in self.comparisons if c.is_measured and c.signed_error_kmh is not None]
        n_meas = len(measured)
        n_unmeas = total - n_meas
        coverage = (n_meas / total) * 100.0
        not_meas_rate = (n_unmeas / total) * 100.0

        if n_meas == 0:
            return RadarBenchmarkSummary(
                total_deliveries=total,
                measured_deliveries=0,
                not_measured_deliveries=n_unmeas,
                coverage_percentage=0.0,
                not_measured_rate=not_meas_rate,
                mae_kmh=0.0,
                rmse_kmh=0.0,
                bias_kmh=0.0,
                within_2kmh_percentage=0.0,
                within_5kmh_percentage=0.0,
                max_error_kmh=0.0,
                min_error_kmh=0.0,
                std_error_kmh=0.0,
                comparisons=self.comparisons
            )

        signed_errors = np.array([c.signed_error_kmh for c in measured], dtype=np.float64)
        abs_errors = np.abs(signed_errors)

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(signed_errors ** 2)))
        bias = float(np.mean(signed_errors))
        w2_pct = float(np.mean(abs_errors <= 2.0) * 100.0)
        w5_pct = float(np.mean(abs_errors <= 5.0) * 100.0)
        max_err = float(np.max(abs_errors))
        min_err = float(np.min(abs_errors))
        std_err = float(np.std(signed_errors))

        return RadarBenchmarkSummary(
            total_deliveries=total,
            measured_deliveries=n_meas,
            not_measured_deliveries=n_unmeas,
            coverage_percentage=coverage,
            not_measured_rate=not_meas_rate,
            mae_kmh=mae,
            rmse_kmh=rmse,
            bias_kmh=bias,
            within_2kmh_percentage=w2_pct,
            within_5kmh_percentage=w5_pct,
            max_error_kmh=max_err,
            min_error_kmh=min_err,
            std_error_kmh=std_err,
            comparisons=self.comparisons
        )

    def generate_plots(
        self, output_image_path: str, title: str = "Cricket Ball Speed - Radar Benchmark Analysis"
    ) -> str:
        """
        Generates and saves the 3-panel benchmark diagnostic dashboard image.
        """
        summary = self.compute_summary()
        return BenchmarkPlotGenerator.generate_dashboard(
            comparisons=self.comparisons,
            summary_dict=summary.to_dict(),
            output_image_path=output_image_path,
            title=title
        )

    def export_full_benchmark_package(
        self, output_dir: str, dataset_name: str = "radar_benchmark"
    ) -> Dict[str, str]:
        """
        Generates complete benchmark suite outputs: JSON, CSV, Markdown, and PNG dashboard.
        """
        os.makedirs(output_dir, exist_ok=True)
        summary = self.compute_summary()

        json_path = os.path.join(output_dir, f"{dataset_name}_report.json")
        csv_path = os.path.join(output_dir, f"{dataset_name}_deliveries.csv")
        md_path = os.path.join(output_dir, f"{dataset_name}_summary.md")
        png_path = os.path.join(output_dir, f"{dataset_name}_plots.png")

        summary.save_json(json_path)
        summary.to_csv(csv_path)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(summary.to_markdown_table())

        self.generate_plots(png_path, title=f"Radar Benchmark - {dataset_name.upper()}")

        return {
            "json_report": json_path,
            "csv_deliveries": csv_path,
            "markdown_summary": md_path,
            "plots_image": png_path,
        }

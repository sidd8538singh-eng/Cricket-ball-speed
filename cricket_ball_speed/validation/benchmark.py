"""
Benchmark Runner: Reproducible Batch Quality-Control and Radar Evaluation Engine.
Processes datasets of deliveries, compares calculated vision speeds against independent radar ground truth,
and produces comprehensive benchmark reports, statistics, and diagnostic plots.

STRICT PRINCIPLE:
Radar speed is strictly an evaluation benchmark; it is NEVER used in calculation pipelines.
"""

from typing import List, Dict, Any, Optional
import os
import json
import csv
import argparse
import logging

from .radar_validation import RadarValidator, RadarBenchmarkSummary, DeliveryRadarComparison
from .quality_control import QualityControlPipeline, QualityControlResult

logger = logging.getLogger("CricketBallSpeed.Benchmark")


class BenchmarkRunner:
    """
    Automated batch evaluation engine across multiple deliveries and datasets.
    """

    def __init__(self, qc_pipeline: Optional[QualityControlPipeline] = None):
        self.qc_pipeline = qc_pipeline or QualityControlPipeline()
        self.radar_validator = RadarValidator()

    def run_radar_evaluation(
        self, deliveries_data: List[Dict[str, Any]]
    ) -> RadarBenchmarkSummary:
        """
        Processes list of deliveries with known radar speeds.
        Each delivery dictionary can contain:
          - "video_name" (or "filename"): str
          - "radar_speed_kmh": float
          - "vision_speed_kmh" (or "calculated_speed"): float or None
          - "c1_c2_speed_kmh": float or None
          - "is_measured": bool (default: vision_speed_kmh is not None)
          - "rejection_reason": str or None
        """
        for item in deliveries_data:
            vid_name = item.get("video_name") or item.get("filename") or "delivery"
            radar_spd = float(item["radar_speed_kmh"])
            vision_spd = item.get("vision_speed_kmh")
            if vision_spd is None:
                vision_spd = item.get("calculated_speed")
            if vision_spd is not None:
                vision_spd = float(vision_spd)

            c1_c2_spd = item.get("c1_c2_speed_kmh")
            if c1_c2_spd is not None:
                c1_c2_spd = float(c1_c2_spd)

            is_meas = item.get("is_measured")
            if is_meas is None:
                is_meas = (vision_spd is not None) and (vision_spd > 0.0)

            rej_reason = item.get("rejection_reason")

            self.radar_validator.add_comparison(
                video_name=vid_name,
                vision_speed_kmh=vision_spd,
                radar_speed_kmh=radar_spd,
                is_measured=bool(is_meas),
                rejection_reason=rej_reason,
                c1_c2_speed_kmh=c1_c2_spd
            )

        return self.radar_validator.compute_summary()

    def evaluate_json_reports_directory(
        self,
        reports_dir: str,
        ground_truth_map: Dict[str, float]
    ) -> RadarBenchmarkSummary:
        """
        Ingests output JSON reports from a directory and matches against ground truth radar mapping.
        ground_truth_map: mapping from video filename or base name to radar speed.
        """
        if not os.path.isdir(reports_dir):
            raise FileNotFoundError(f"Reports directory not found: {reports_dir}")

        deliveries: List[Dict[str, Any]] = []

        for fname in os.listdir(reports_dir):
            if fname.endswith(".json"):
                full_path = os.path.join(reports_dir, fname)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    meta = data.get("video_metadata", {})
                    v_name = meta.get("file_name") or os.path.splitext(fname)[0] + ".mp4"
                    base_k = os.path.splitext(v_name)[0]

                    # Find matching radar speed
                    radar_spd = ground_truth_map.get(v_name) or ground_truth_map.get(base_k)
                    if radar_spd is None:
                        # Check inside JSON extra_info
                        radar_spd = data.get("extra_info", {}).get("radar_speed")

                    if radar_spd is None:
                        logger.warning(f"Skipping {v_name}: No ground truth radar speed found.")
                        continue

                    speed_meas = data.get("speed_measurement", {})
                    rel_res = speed_meas.get("release_speed") or {}
                    c1c2_res = speed_meas.get("c1_c2_speed") or {}
                    qc = data.get("quality_control", {})

                    is_meas = bool(qc.get("is_measured", False))
                    v_spd = rel_res.get("speed_km_h") if is_meas else None
                    c1c2_spd = c1c2_res.get("average_speed_km_h") if c1c2_res else None
                    rej = qc.get("rejection_reason")

                    deliveries.append({
                        "video_name": v_name,
                        "radar_speed_kmh": float(radar_spd),
                        "vision_speed_kmh": v_spd,
                        "c1_c2_speed_kmh": c1c2_spd,
                        "is_measured": is_meas,
                        "rejection_reason": rej,
                    })
                except Exception as e:
                    logger.error(f"Failed to read report {fname}: {e}")

        return self.run_radar_evaluation(deliveries)

    def export_benchmark_results(
        self, output_dir: str, dataset_name: str = "radar_benchmark"
    ) -> Dict[str, str]:
        """Exports JSON, CSV, Markdown, and PNG plots."""
        return self.radar_validator.export_full_benchmark_package(
            output_dir=output_dir, dataset_name=dataset_name
        )


def load_ground_truth(path: str) -> Dict[str, float]:
    """Loads ground truth radar speeds from JSON or CSV file."""
    if not os.path.isfile(path):
        return {}

    mapping: Dict[str, float] = {}
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for k, v in data.items():
                mapping[k] = float(v)
    elif path.endswith(".csv"):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = row.get("video_name") or row.get("filename")
                v = row.get("radar_speed_kmh") or row.get("radar_speed")
                if k and v:
                    mapping[k] = float(v)
    return mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cricket Bowling Speed - Radar Benchmark Evaluator")
    parser.add_argument("--reports-dir", type=str, required=True, help="Directory containing delivery JSON reports")
    parser.add_argument("--ground-truth", type=str, required=True, help="JSON or CSV file with radar speeds")
    parser.add_argument("--output-dir", type=str, default="storage/benchmark", help="Directory to save benchmark reports")
    parser.add_argument("--dataset-name", type=str, default="radar_evaluation", help="Name prefix for output files")
    args = parser.parse_args()

    gt = load_ground_truth(args.ground_truth)
    runner = BenchmarkRunner()
    summary = runner.evaluate_json_reports_directory(args.reports_dir, gt)
    paths = runner.export_benchmark_results(args.output_dir, args.dataset_name)

    print(f"\n{summary.to_markdown_table()}")
    print("\nBenchmark reports generated:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

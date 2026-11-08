"""
Benchmark Plots Generator: Renders diagnostic accuracy visualizations.
Generates:
  1. Predicted vs Radar Speed scatter plot with 1:1 identity line and +-2 km/h / +-5 km/h corridors.
  2. Residuals (Signed Error) vs Radar Speed plot.
  3. Error Distribution (Histogram and density curve) with Bias indicator.
  4. Executive Performance Metric Summary Card.

IMPLEMENTATION:
Supports matplotlib if installed; provides an ultra-reliable, zero-dependency OpenCV renderer
as an automatic fallback so plots can ALWAYS be generated in any headless or restricted environment.
"""

from typing import List, Optional, Tuple, Dict, Any
import os
import math
import numpy as np
import cv2

# Check for matplotlib
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


class BenchmarkPlotGenerator:
    """
    Renders publication-quality diagnostic plots for radar benchmark evaluations.
    """

    @classmethod
    def generate_dashboard(
        cls,
        comparisons: List[Any],
        summary_dict: Dict[str, Any],
        output_image_path: str,
        title: str = "Cricket Ball Speed - Radar Benchmark Analysis"
    ) -> str:
        """
        Renders the complete 3-panel benchmark dashboard and saves to output_image_path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)

        if MATPLOTLIB_AVAILABLE:
            try:
                return cls._render_matplotlib(comparisons, summary_dict, output_image_path, title)
            except Exception as e:
                # Fallback to OpenCV if matplotlib has rendering or backend issues
                pass

        return cls._render_opencv(comparisons, summary_dict, output_image_path, title)

    @classmethod
    def _render_matplotlib(
        cls,
        comparisons: List[Any],
        summary: Dict[str, Any],
        output_path: str,
        title: str
    ) -> str:
        """Renders dashboard using matplotlib."""
        measured = [c for c in comparisons if c.is_measured and c.vision_speed_kmh is not None]
        radar_vals = [c.radar_speed_kmh for c in measured]
        calc_vals = [c.vision_speed_kmh for c in measured]
        errors = [c.error_kmh for c in measured]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

        # Panel 1: Predicted vs Radar
        if radar_vals:
            min_v = max(20.0, min(radar_vals + calc_vals) - 5.0)
            max_v = min(180.0, max(radar_vals + calc_vals) + 5.0)
            x_line = np.linspace(min_v, max_v, 100)

            ax1.plot(x_line, x_line, "k--", label="1:1 Perfect Agreement", alpha=0.7)
            ax1.fill_between(x_line, x_line - 2.0, x_line + 2.0, color="green", alpha=0.15, label="±2.0 km/h Corridor")
            ax1.fill_between(x_line, x_line - 5.0, x_line + 5.0, color="gold", alpha=0.08, label="±5.0 km/h Corridor")
            ax1.scatter(radar_vals, calc_vals, color="#1f77b4", edgecolor="black", s=50, zorder=5)

            ax1.set_xlim(min_v, max_v)
            ax1.set_ylim(min_v, max_v)
        ax1.set_title("Calculated vs Radar Speed")
        ax1.set_xlabel("Radar Speed (km/h)")
        ax1.set_ylabel("Calculated 3D Release Speed (km/h)")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left", fontsize=9)

        # Panel 2: Residuals vs Radar
        if radar_vals:
            ax2.axhline(0, color="k", linestyle="--", alpha=0.7, label="Zero Error")
            ax2.axhline(2.0, color="green", linestyle=":", alpha=0.8, label="+2.0 km/h")
            ax2.axhline(-2.0, color="green", linestyle=":", alpha=0.8, label="-2.0 km/h")
            ax2.axhline(5.0, color="orange", linestyle=":", alpha=0.6)
            ax2.axhline(-5.0, color="orange", linestyle=":", alpha=0.6)
            ax2.scatter(radar_vals, errors, color="#d62728", edgecolor="black", s=50, zorder=5)
            ax2.set_xlim(min_v, max_v)
            max_err = max(6.0, max(abs(e) for e in errors) + 2.0)
            ax2.set_ylim(-max_err, max_err)
        ax2.set_title("Residuals (Calculated - Radar)")
        ax2.set_xlabel("Radar Speed (km/h)")
        ax2.set_ylabel("Signed Error (km/h)")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper left", fontsize=9)

        # Panel 3: Error Distribution Histogram
        if errors:
            bins = np.linspace(-6, 6, 13)
            ax3.hist(errors, bins=bins, color="#2ca02c", edgecolor="black", alpha=0.7, density=True, label="Errors")
            ax3.axvline(summary.get("bias_kmh", 0.0), color="blue", linestyle="-", linewidth=2, label=f"Bias: {summary.get('bias_kmh', 0.0):.2f} km/h")
            ax3.axvline(2.0, color="green", linestyle=":", alpha=0.8)
            ax3.axvline(-2.0, color="green", linestyle=":", alpha=0.8)
        ax3.set_title("Signed Error Distribution")
        ax3.set_xlabel("Error (km/h)")
        ax3.set_ylabel("Density")
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(loc="upper left", fontsize=9)

        # Summary box
        stats_text = (
            f"Deliveries: {summary.get('total_deliveries', 0)} (Coverage: {summary.get('coverage_percentage', 0):.1f}%)\n"
            f"MAE: {summary.get('mae_kmh', 0):.2f} km/h | RMSE: {summary.get('rmse_kmh', 0):.2f} km/h\n"
            f"Within ±2 km/h: {summary.get('within_2kmh_percentage', 0):.1f}% | ±5 km/h: {summary.get('within_5kmh_percentage', 0):.1f}%"
        )
        fig.text(0.5, -0.05, stats_text, ha="center", fontsize=11, bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        return output_path

    @classmethod
    def _render_opencv(
        cls,
        comparisons: List[Any],
        summary: Dict[str, Any],
        output_path: str,
        title: str
    ) -> str:
        """
        Pure OpenCV renderer. Zero external graphical dependencies.
        Produces an ultra-clean 1800x1000 PNG benchmark dashboard.
        """
        width = 1800
        height = 1000
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 248  # Crisp off-white background

        # Header Banner
        cv2.rectangle(canvas, (0, 0), (width, 85), (28, 38, 48), -1)
        cv2.putText(
            canvas, title, (40, 42),
            cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            "Independent Doppler Radar Ground Truth Verification | Automated Diagnostic Dashboard",
            (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 200, 220), 1, cv2.LINE_AA
        )

        # Extract measured data
        measured = [c for c in comparisons if getattr(c, "is_measured", False) and getattr(c, "vision_speed_kmh", None) is not None]
        radar_vals = [float(c.radar_speed_kmh) for c in measured]
        calc_vals = [float(c.vision_speed_kmh) for c in measured]
        errors = [float(c.error_kmh) for c in measured]

        # Panel coordinates
        top_y = 110
        panel_h = 660
        gap = 30
        panel_w = (width - 4 * gap) // 3

        # =====================================================================
        # Panel 1: Calculated vs Radar Speed (Scatter)
        # =====================================================================
        p1_x = gap
        cls._draw_panel_box(canvas, p1_x, top_y, panel_w, panel_h, "1. Calculated vs Radar Speed")
        plot_x1 = p1_x + 65
        plot_y1 = top_y + 60
        plot_w1 = panel_w - 90
        plot_h1 = panel_h - 120

        min_s = 60.0
        max_s = 160.0
        if radar_vals:
            all_s = radar_vals + calc_vals
            min_s = max(20.0, math.floor(min(all_s) / 10.0) * 10.0 - 5.0)
            max_s = min(175.0, math.ceil(max(all_s) / 10.0) * 10.0 + 5.0)

        cls._draw_axes(canvas, plot_x1, plot_y1, plot_w1, plot_h1, min_s, max_s, min_s, max_s, "Radar (km/h)", "Calculated (km/h)")

        # Corridors: +-5 km/h (light yellow) & +-2 km/h (light green)
        cls._draw_corridor_band(canvas, plot_x1, plot_y1, plot_w1, plot_h1, min_s, max_s, 5.0, (210, 245, 255))
        cls._draw_corridor_band(canvas, plot_x1, plot_y1, plot_w1, plot_h1, min_s, max_s, 2.0, (215, 245, 215))

        # 1:1 Identity Line
        pt_start = cls._map_point(min_s, min_s, plot_x1, plot_y1, plot_w1, plot_h1, min_s, max_s, min_s, max_s)
        pt_end = cls._map_point(max_s, max_s, plot_x1, plot_y1, plot_w1, plot_h1, min_s, max_s, min_s, max_s)
        cv2.line(canvas, pt_start, pt_end, (80, 80, 80), 2, cv2.LINE_AA)

        # Plot points
        for r, c, err in zip(radar_vals, calc_vals, errors):
            pt = cls._map_point(r, c, plot_x1, plot_y1, plot_w1, plot_h1, min_s, max_s, min_s, max_s)
            color = (34, 139, 34) if abs(err) <= 2.0 else ((0, 165, 255) if abs(err) <= 5.0 else (0, 0, 220))
            cv2.circle(canvas, pt, 6, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 7, (0, 0, 0), 1, cv2.LINE_AA)

        # =====================================================================
        # Panel 2: Residuals vs Radar Speed
        # =====================================================================
        p2_x = p1_x + panel_w + gap
        cls._draw_panel_box(canvas, p2_x, top_y, panel_w, panel_h, "2. Residuals (Calculated - Radar)")
        plot_x2 = p2_x + 65
        plot_y2 = top_y + 60
        plot_w2 = panel_w - 90
        plot_h2 = panel_h - 120

        max_abs_e = 6.0
        if errors:
            max_abs_e = max(6.0, math.ceil(max(abs(e) for e in errors) + 1.0))

        cls._draw_axes(canvas, plot_x2, plot_y2, plot_w2, plot_h2, min_s, max_s, -max_abs_e, max_abs_e, "Radar (km/h)", "Signed Error (km/h)")

        # Tolerance horizontal lines
        cls._draw_hband(canvas, plot_x2, plot_y2, plot_w2, plot_h2, -max_abs_e, max_abs_e, 2.0, (215, 245, 215))
        cls._draw_hline(canvas, plot_x2, plot_y2, plot_w2, plot_h2, -max_abs_e, max_abs_e, 0.0, (80, 80, 80), 2)
        cls._draw_hline(canvas, plot_x2, plot_y2, plot_w2, plot_h2, -max_abs_e, max_abs_e, 2.0, (34, 139, 34), 1)
        cls._draw_hline(canvas, plot_x2, plot_y2, plot_w2, plot_h2, -max_abs_e, max_abs_e, -2.0, (34, 139, 34), 1)

        # Plot error points
        for r, err in zip(radar_vals, errors):
            pt = cls._map_point(r, err, plot_x2, plot_y2, plot_w2, plot_h2, min_s, max_s, -max_abs_e, max_abs_e)
            color = (34, 139, 34) if abs(err) <= 2.0 else ((0, 165, 255) if abs(err) <= 5.0 else (0, 0, 220))
            cv2.circle(canvas, pt, 6, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 7, (0, 0, 0), 1, cv2.LINE_AA)

        # =====================================================================
        # Panel 3: Error Distribution Histogram
        # =====================================================================
        p3_x = p2_x + panel_w + gap
        cls._draw_panel_box(canvas, p3_x, top_y, panel_w, panel_h, "3. Signed Error Distribution")
        plot_x3 = p3_x + 65
        plot_y3 = top_y + 60
        plot_w3 = panel_w - 90
        plot_h3 = panel_h - 120

        cls._draw_histogram(canvas, plot_x3, plot_y3, plot_w3, plot_h3, errors, -max_abs_e, max_abs_e)

        # =====================================================================
        # Bottom Summary Metric Card
        # =====================================================================
        cls._draw_metric_card(canvas, 30, height - 190, width - 60, 160, summary)

        cv2.imwrite(output_path, canvas)
        return output_path

    @classmethod
    def _draw_panel_box(cls, canvas, x, y, w, h, title):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (210, 215, 220), 1)
        cv2.rectangle(canvas, (x, y), (x + w, y + 40), (240, 244, 248), -1)
        cv2.line(canvas, (x, y + 40), (x + w, y + 40), (210, 215, 220), 1)
        cv2.putText(canvas, title, (x + 18, y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 40, 50), 2, cv2.LINE_AA)

    @classmethod
    def _draw_axes(cls, canvas, px, py, pw, ph, xmin, xmax, ymin, ymax, xlabel, ylabel):
        # Background grid
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (250, 250, 250), -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (180, 185, 190), 1)

        # X grid lines and labels
        n_ticks_x = 5
        for i in range(n_ticks_x + 1):
            val = xmin + i * (xmax - xmin) / n_ticks_x
            x_pos = int(px + i * pw / n_ticks_x)
            cv2.line(canvas, (x_pos, py), (x_pos, py + ph), (230, 230, 230), 1)
            cv2.putText(canvas, f"{val:.0f}", (x_pos - 12, py + ph + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

        # Y grid lines and labels
        n_ticks_y = 5
        for i in range(n_ticks_y + 1):
            val = ymin + i * (ymax - ymin) / n_ticks_y
            y_pos = int(py + ph - i * ph / n_ticks_y)
            cv2.line(canvas, (px, y_pos), (px + pw, y_pos), (230, 230, 230), 1)
            cv2.putText(canvas, f"{val:.1f}" if abs(val) < 10 else f"{val:.0f}", (px - 45, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)

        # Axis labels
        cv2.putText(canvas, xlabel, (px + pw // 2 - 40, py + ph + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 50, 60), 1, cv2.LINE_AA)
        cv2.putText(canvas, ylabel, (px - 55, py - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 50, 60), 1, cv2.LINE_AA)

    @classmethod
    def _map_point(cls, x, y, px, py, pw, ph, xmin, xmax, ymin, ymax) -> Tuple[int, int]:
        norm_x = (x - xmin) / max(1e-5, (xmax - xmin))
        norm_y = (y - ymin) / max(1e-5, (ymax - ymin))
        pt_x = int(px + norm_x * pw)
        pt_y = int(py + (1.0 - norm_y) * ph)
        return (pt_x, pt_y)

    @classmethod
    def _draw_corridor_band(cls, canvas, px, py, pw, ph, smin, smax, corridor, color):
        pts = [
            cls._map_point(smin, smin - corridor, px, py, pw, ph, smin, smax, smin, smax),
            cls._map_point(smax, smax - corridor, px, py, pw, ph, smin, smax, smin, smax),
            cls._map_point(smax, smax + corridor, px, py, pw, ph, smin, smax, smin, smax),
            cls._map_point(smin, smin + corridor, px, py, pw, ph, smin, smax, smin, smax),
        ]
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], color)
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    @classmethod
    def _draw_hband(cls, canvas, px, py, pw, ph, ymin, ymax, bound, color):
        p1 = cls._map_point(0, -bound, px, py, pw, ph, 0, 1, ymin, ymax)
        p2 = cls._map_point(1, bound, px, py, pw, ph, 0, 1, ymin, ymax)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (px, min(p1[1], p2[1])), (px + pw, max(p1[1], p2[1])), color, -1)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

    @classmethod
    def _draw_hline(cls, canvas, px, py, pw, ph, ymin, ymax, yval, color, thickness):
        pt1 = cls._map_point(0, yval, px, py, pw, ph, 0, 1, ymin, ymax)
        cv2.line(canvas, (px, pt1[1]), (px + pw, pt1[1]), color, thickness, cv2.LINE_AA)

    @classmethod
    def _draw_histogram(cls, canvas, px, py, pw, ph, errors, emin, emax):
        cls._draw_axes(canvas, px, py, pw, ph, emin, emax, 0, 10, "Signed Error (km/h)", "Deliveries Count")
        if not errors:
            cv2.putText(canvas, "No measured delivery errors", (px + pw // 4, py + ph // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
            return

        n_bins = 11
        bin_edges = np.linspace(emin, emax, n_bins + 1)
        counts, _ = np.histogram(errors, bins=bin_edges)
        max_count = max(5, int(np.max(counts)) + 2)

        # Redraw Y axis with max_count
        cls._draw_axes(canvas, px, py, pw, ph, emin, emax, 0, max_count, "Signed Error (km/h)", "Deliveries Count")

        # Zero line & +-2km/h lines
        z_pt = cls._map_point(0, 0, px, py, pw, ph, emin, emax, 0, max_count)
        cv2.line(canvas, (z_pt[0], py), (z_pt[0], py + ph), (80, 80, 80), 2, cv2.LINE_AA)
        p2_pt = cls._map_point(2.0, 0, px, py, pw, ph, emin, emax, 0, max_count)
        cv2.line(canvas, (p2_pt[0], py), (p2_pt[0], py + ph), (34, 139, 34), 1, cv2.LINE_AA)
        m2_pt = cls._map_point(-2.0, 0, px, py, pw, ph, emin, emax, 0, max_count)
        cv2.line(canvas, (m2_pt[0], py), (m2_pt[0], py + ph), (34, 139, 34), 1, cv2.LINE_AA)

        # Draw bars
        for i in range(n_bins):
            b_left = bin_edges[i]
            b_right = bin_edges[i + 1]
            c = counts[i]
            if c > 0:
                p_tl = cls._map_point(b_left, c, px, py, pw, ph, emin, emax, 0, max_count)
                p_br = cls._map_point(b_right, 0, px, py, pw, ph, emin, emax, 0, max_count)
                mid_val = (b_left + b_right) / 2.0
                bar_col = (76, 175, 80) if abs(mid_val) <= 2.0 else ((33, 150, 243) if abs(mid_val) <= 5.0 else (244, 67, 54))
                cv2.rectangle(canvas, (p_tl[0] + 1, p_tl[1]), (p_br[0] - 1, p_br[1]), bar_col, -1)
                cv2.rectangle(canvas, (p_tl[0] + 1, p_tl[1]), (p_br[0] - 1, p_br[1]), (50, 50, 50), 1)

    @classmethod
    def _draw_metric_card(cls, canvas, x, y, w, h, summary):
        # Card background
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (210, 215, 220), 1)

        # Title bar
        cv2.rectangle(canvas, (x, y), (x + w, y + 36), (235, 240, 245), -1)
        cv2.line(canvas, (x, y + 36), (x + w, y + 36), (210, 215, 220), 1)
        cv2.putText(canvas, "EXECUTIVE RADAR BENCHMARK METRICS & KPI AUDIT", (x + 20, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 40, 50), 2, cv2.LINE_AA)

        # 6 KPI Tiles
        n_tiles = 6
        tile_w = (w - 40) // n_tiles
        tile_y = y + 50

        kpis = [
            ("TOTAL DELIVERIES", f"{summary.get('total_deliveries', 0)}", f"{summary.get('measured_deliveries', 0)} Measured | {summary.get('not_measured_deliveries', 0)} Rejected", (40, 50, 60)),
            ("MEASUREMENT COVERAGE", f"{summary.get('coverage_percentage', 0):.1f}%", f"NOT_MEASURED Rate: {summary.get('not_measured_rate', 0):.1f}%", (34, 139, 34) if summary.get('coverage_percentage', 0) >= 80 else (0, 140, 255)),
            ("MEAN ABS ERROR (MAE)", f"{summary.get('mae_kmh', 0):.2f} km/h", "Average Absolute Delta", (40, 50, 60)),
            ("ROOT MEAN SQ ERROR (RMSE)", f"{summary.get('rmse_kmh', 0):.2f} km/h", "System Variance Penalty", (40, 50, 60)),
            ("SYSTEM BIAS", f"{summary.get('bias_kmh', 0):+.2f} km/h", "Mean Signed Calibration Drift", (34, 139, 34) if abs(summary.get('bias_kmh', 0)) <= 1.0 else (0, 140, 255)),
            ("ACCURACY IN ±2.0 km/h", f"{summary.get('within_2kmh_percentage', 0):.1f}%", f"In ±5.0 km/h: {summary.get('within_5kmh_percentage', 0):.1f}%", (34, 139, 34) if summary.get('within_2kmh_percentage', 0) >= 90 else (0, 140, 255)),
        ]

        for i, (kpi_label, kpi_val, kpi_sub, col) in enumerate(kpis):
            tx = x + 20 + i * tile_w
            # Tile separator
            if i > 0:
                cv2.line(canvas, (tx - 10, tile_y), (tx - 10, tile_y + 85), (225, 230, 235), 1)

            cv2.putText(canvas, kpi_label, (tx, tile_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (110, 120, 130), 1, cv2.LINE_AA)
            cv2.putText(canvas, kpi_val, (tx, tile_y + 48), cv2.FONT_HERSHEY_DUPLEX, 0.95, col, 2, cv2.LINE_AA)
            cv2.putText(canvas, kpi_sub, (tx, tile_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 130, 140), 1, cv2.LINE_AA)

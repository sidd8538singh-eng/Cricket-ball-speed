"""
Master Orchestrator Entry Point: Zoota-Style Cricket Ball Speed Analyzer.
Supports both:
  1. FastAPI Server (Live Camera WebSocket + REST API + Web Frontend)
  2. Standalone CLI delivery analysis pipeline
"""

import sys
import os
import tempfile
import argparse
from typing import Optional

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
os.environ["YOLO_CONFIG_DIR"] = tempfile.gettempdir()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CBS_DIR = os.path.join(CURRENT_DIR, "cricket_ball_speed")
BACKEND_DIR = os.path.join(CURRENT_DIR, "backend")

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if CBS_DIR not in sys.path:
    sys.path.insert(0, CBS_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def run_server(host: Optional[str] = None, port: Optional[int] = None, reload: bool = False):
    """Starts the FastAPI Web and WebSocket server."""
    import uvicorn
    server_host = host or os.getenv("HOST", "0.0.0.0")
    server_port = port if port is not None else int(os.getenv("PORT", "8000"))
    print("=" * 70)
    print("STARTING CRICKET BOWLING SPEED ANALYZER SERVER")
    print(f"Server URL:     http://{server_host}:{server_port}")
    print(f"API Docs:       http://{server_host}:{server_port}/docs")
    print(f"Healthcheck:    http://{server_host}:{server_port}/health")
    print(f"Live WebSocket: ws://{server_host}:{server_port}/ws/live")
    print("=" * 70)
    uvicorn.run("backend.app:app", host=server_host, port=server_port, reload=reload)


def main():
    parser = argparse.ArgumentParser(
        description="Zoota-Style Cricket Bowling Speed Measurement Platform",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Mode to execute")

    # Subcommand: server
    server_parser = subparsers.add_parser("server", help="Start the FastAPI backend server and Web UI")
    server_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    server_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    # Subcommand: analyze (CLI)
    cli_parser = subparsers.add_parser("analyze", help="Run CLI bowling speed pipeline on a recorded video")
    cli_parser.add_argument("--video", type=str, required=True, help="Path to cricket video file")
    cli_parser.add_argument("--calibration", type=str, default=None, help="Path to locked calibration JSON")
    cli_parser.add_argument("--calibrate-only", action="store_true", help="Calibrate camera and exit")
    cli_parser.add_argument("--headless", action="store_true", help="Run in headless mode without UI prompts")
    cli_parser.add_argument("--force-manual", action="store_true", help="Force interactive manual 8-point calibration")
    cli_parser.add_argument("--radar-speed", type=float, default=None, help="Independent radar gun speed for benchmark comparison")
    cli_parser.add_argument("--output-dir", type=str, default="storage/results", help="Directory for output reports and videos")
    cli_parser.add_argument("--no-video-output", action="store_true", help="Skip rendering output MP4 videos")

    # Top-level direct arguments for compatibility
    parser.add_argument("--server", action="store_true", help="Shortcut to start backend server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--video", type=str, default=None, help="Video path for direct CLI analysis")
    parser.add_argument("--calibration", type=str, default=None)
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--force-manual", action="store_true")
    parser.add_argument("--radar-speed", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="storage/results")
    parser.add_argument("--no-video-output", action="store_true")

    args = parser.parse_args()

    # Route command
    if args.server or args.command == "server":
        run_server(host=args.host, port=args.port, reload=getattr(args, "reload", False))
        return

    video_path = getattr(args, "video", None)
    if video_path:
        try:
            from cricket_ball_speed.main import CricketSpeedPipeline
        except ImportError:
            from main import CricketSpeedPipeline
        pipeline = CricketSpeedPipeline(config_dir=os.path.join(CBS_DIR, "config"))
        res = pipeline.process_delivery(
            video_path=video_path,
            calibration_path=args.calibration,
            calibrate_only=args.calibrate_only,
            headless=args.headless,
            force_manual=args.force_manual,
            radar_speed=args.radar_speed,
            output_dir=args.output_dir,
            generate_videos=not args.no_video_output
        )

        print("\n" + "=" * 65)
        print("CRICKET BOWLING SPEED MEASUREMENT SUMMARY")
        print("=" * 65)
        print(f"Status:             {res.get('status')}")
        if res.get("status") == "MEASURED":
            print(f"Release Speed:      {res.get('release_speed_kmh'):.1f} km/h (+- {res.get('uncertainty_kmh'):.1f} km/h)")
            if res.get("c1_c2_speed_kmh"):
                print(f"C1-C2 Avg Speed:    {res.get('c1_c2_speed_kmh'):.1f} km/h")
            print(f"Quality Score:      {res.get('qc_score'):.0f} / 100")
        else:
            print(f"Rejection Reason:   {res.get('rejection_reason')}")

        print(f"JSON Report:        {res.get('json_report')}")
        print(f"Excel Report:       {res.get('excel_report')}")
        if res.get("final_video"):
            print(f"Final Video:        {res.get('final_video')}")
        if res.get("diagnostic_video"):
            print(f"Diagnostic Video:   {res.get('diagnostic_video')}")
        print("=" * 65 + "\n")
        return

    # Default to starting server if no CLI arguments passed
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

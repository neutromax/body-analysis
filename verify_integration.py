"""
Auto Annotator — End-to-End Integration Verification Script
=============================================================

Tests the full pipeline: yt-dlp URL resolution → PyAV in-memory decoding →
MediaPipe pose detection → calibration → classification → aggregation → Excel.

No screen capture, no GUI, no window focus required.
"""

import os
import sys
import time
import shutil
from pathlib import Path

# Add auto_annotator directory to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "auto_annotator"))

import auto_config
from stream_reader import StreamReader
from pose_engine import PoseEngine
from feature_classifier import FrameMetrics, compute_frame_metrics, classify_frame
from calibration import Calibrator
from aggregator import aggregate_window
from excel_writer import ExcelWriter
import openpyxl


def run_verification():
    print("=" * 60)
    print("  AUTO ANNOTATOR — IN-MEMORY STREAM E2E TEST")
    print("=" * 60)

    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    # 1. Setup temporary Excel output file
    temp_dir = Path(__file__).parent / "temp_test"
    temp_dir.mkdir(exist_ok=True)
    excel_path = temp_dir / "test_output.xlsx"
    if excel_path.exists():
        excel_path.unlink()

    print(f"\n[1/6] Excel output: {excel_path.resolve()}")

    # 2. Start the stream reader
    print(f"[2/6] Resolving YouTube URL and opening PyAV stream...")
    reader = StreamReader(youtube_url=youtube_url)
    reader.start()

    # Wait for the stream to resolve and first frame to be decoded
    start_wait = time.time()
    while time.time() - start_wait < 30:
        frame = reader.current_frame
        if frame is not None:
            print(f"      First frame decoded! Shape: {frame.shape}, PTS: {reader.current_pts:.2f}s")
            break
        time.sleep(0.5)
    else:
        print("      FAILED: Timed out waiting for first decoded frame.")
        reader.stop()
        sys.exit(1)

    # 3. Initialize pose engine
    print("[3/6] Initializing MediaPipe PoseEngine...")
    engine = PoseEngine()

    # 4. Calibration phase — collect 3 frames
    print("[4/6] Running calibration (3 frames)...")
    calibrator = Calibrator()
    auto_config.CALIBRATION_DURATION = 3
    prev_metrics = None

    for i in range(3):
        time.sleep(1.0)  # Wait 1 second between frames
        frame = reader.current_frame
        if frame is None:
            print(f"      Frame {i+1}: No frame available, skipping.")
            continue

        landmarks = engine.detect(frame)
        if landmarks is None:
            print(f"      Frame {i+1}: No pose detected (PTS={reader.current_pts:.2f}s)")
            metrics = FrameMetrics()
        else:
            metrics = compute_frame_metrics(landmarks, prev_metrics)
            prev_metrics = metrics
            print(f"      Frame {i+1}: Pose detected! angle={metrics.body_angle_deg}, "
                  f"head_x={metrics.head_local_x} (PTS={reader.current_pts:.2f}s)")

        calibrator.add_reading(metrics)

    thresholds = calibrator.compute_thresholds()
    print(f"      Thresholds: body={thresholds.T_body:.2f}, head={thresholds.T_head:.4f}, "
          f"hand={thresholds.T_hand:.4f}, leg={thresholds.T_leg:.4f}, "
          f"calibrated={thresholds.calibrated}")

    # 5. Annotation phase — collect frames for one 10-second window
    print("[5/6] Running annotation (10 frames for 1 window)...")
    auto_config.OBSERVATIONS_PER_WINDOW = 10
    window_readings = []

    for i in range(10):
        time.sleep(0.5)  # Faster for testing
        frame = reader.current_frame
        if frame is None:
            print(f"      Tick {i+1}: No frame.")
            window_readings.append({f: None for f in ["hand", "leg", "head", "body"]})
            continue

        landmarks = engine.detect(frame)
        if landmarks is None:
            print(f"      Tick {i+1}: No pose (PTS={reader.current_pts:.2f}s)")
            window_readings.append({f: None for f in ["hand", "leg", "head", "body"]})
        else:
            metrics = compute_frame_metrics(landmarks, prev_metrics)
            prev_metrics = metrics
            reading = classify_frame(metrics, thresholds)
            window_readings.append(reading)
            print(f"      Tick {i+1}: hand={reading['hand']}, leg={reading['leg']}, "
                  f"head={reading['head']}, body={reading['body']} (PTS={reader.current_pts:.2f}s)")

    # Aggregate
    result = aggregate_window(window_readings)
    print(f"      Aggregated: {result}")

    # Write to Excel
    writer = ExcelWriter(str(excel_path))
    writer.start_video(youtube_url)
    writer.append_row("0:00 – 0:10", result)

    # 6. Cleanup and verify
    print("[6/6] Verifying Excel output...")
    reader.stop()
    engine.close()

    wb = openpyxl.load_workbook(str(excel_path))
    sheet = wb.active
    rows = list(sheet.iter_rows(values_only=True))

    print(f"      Header: {rows[0]}")
    assert rows[0] == ("Video Link", "Time Window", "Hand", "Leg", "Head", "Body"), \
        "Header mismatch!"

    for row in rows[1:]:
        print(f"      Data:   {row}")

    assert len(rows) > 1, "No data rows written!"

    # Cleanup
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass

    print()
    print("=" * 60)
    print("  ALL CHECKS PASSED — IN-MEMORY PIPELINE WORKING!")
    print("=" * 60)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_verification()

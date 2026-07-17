"""
Auto Annotator — End-to-End Integration Verification Script
=============================================================

This script runs the entire video capture, MediaPipe pose detection, calibration,
classification, aggregation, and Excel writing pipeline programmatically without
launching the full Tkinter GUI.
"""

import os
import sys
import time
import shutil
from pathlib import Path

# Add auto_annotator directory to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "auto_annotator"))

import auto_config
from webview_player import WebViewPlayer
from session_controller import SessionController, SessionState
import openpyxl


def run_verification():
    print("=== AUTO ANNOTATOR INTEGRATION TEST ===")
    
    # 1. Temporarily patch config constants for quick test execution
    print("Patching window constants for quick evaluation...")
    auto_config.WINDOW_DURATION = 3
    auto_config.OBSERVATIONS_PER_WINDOW = 3
    auto_config.CALIBRATION_DURATION = 3
    
    # 2. Setup temporary Excel output file
    temp_dir = Path(__file__).parent / "temp_test"
    temp_dir.mkdir(exist_ok=True)
    excel_path = temp_dir / "test_output.xlsx"
    if excel_path.exists():
        excel_path.unlink()
        
    print(f"Excel target output: {excel_path.resolve()}")

    # 3. Load webview player with an embeddable video
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    print(f"Loading WebViewPlayer with video: {youtube_url}")
    player = WebViewPlayer()
    player.load(youtube_url)

    # 4. Wait for player to load and become ready
    print("Waiting for YouTube player to finish handshakes and enter ready state...")
    ready = False
    start_wait = time.time()
    while time.time() - start_wait < 35:  # Allow extra time for first-run downloads
        status = player.get_status()
        print(f"  Current status: {status} (elapsed: {int(time.time() - start_wait)}s)")
        if status == "ready":
            ready = True
            break
        elif "error" in status:
            print(f"  FAILED: Player encountered error: {status}")
            player.close()
            sys.exit(1)
        time.sleep(1.0)

    if not ready:
        print("  FAILED: Player timed out waiting to become ready.")
        player.close()
        sys.exit(1)

    print("Player is READY. Launching SessionController...")

    # 5. Initialize controller and start the session
    # Mock gui callback to print status to stdout
    def gui_callback(status):
        print(f"  [GUI Callback] State: {status.get('state')} | Window: {status.get('time_window')} | Time: {status.get('current_time')}s")

    # Define a dummy root.after mechanism for ticking
    ticks_fired = []
    
    controller = SessionController(
        player=player,
        excel_path=str(excel_path),
        video_url=youtube_url,
        root_after=lambda ms, cb: ticks_fired.append(cb),
        gui_callback=gui_callback,
    )
    
    # Start session (triggers calibration)
    controller.start()
    
    # 6. Execute ticks manually
    print("\nExecuting calibration & annotation loop...")
    # Ticks:
    # Ticks 1-3: Calibration
    # Ticks 4-6: Running annotation (completing one 3-second window)
    # Tick 7: Buffer tick
    for tick_num in range(1, 9):
        print(f"\n--- Tick {tick_num} ---")
        if not ticks_fired:
            print("  Error: No tick callback scheduled by controller.")
            break
            
        cb = ticks_fired.pop(0)
        
        # We need to make sure the player registers as playing so ticks are processed
        # During headless execution, player state in shared memory might be unstarted or cued (5).
        # We temporarily force state to playing (1) in shared state to allow execution.
        player._shared_state["state"] = 1
        
        # Execute the controller tick callback
        cb()
        
        # Verify state changes
        if tick_num == 3:
            print(f"  Check state after calibration: {controller.state} (Expected: SessionState.RUNNING)")
            assert controller.state == SessionState.RUNNING, "Session should have transitioned to RUNNING."
            print(f"  Calibrated thresholds: {controller._thresholds}")
            assert controller._thresholds is not None, "Thresholds must be computed after calibration."
            
        time.sleep(0.5)

    print("\nStopping annotation session...")
    controller.stop()
    player.close()
    
    # 7. Verify Excel file contents
    print("\nVerifying Excel output file...")
    assert excel_path.exists(), "Excel file was not created!"
    
    wb = openpyxl.load_workbook(str(excel_path))
    sheet = wb.active
    
    rows = list(sheet.iter_rows(values_only=True))
    print(f"Header Row: {rows[0]}")
    assert rows[0] == ("Video Link", "Time Window", "Hand", "Leg", "Head", "Body"), "Excel header columns do not match specification!"
    
    print("Written Rows:")
    for row in rows[1:]:
        print(f"  {row}")
        
    assert len(rows) > 1, "No data rows were written to the Excel file!"
    
    # Clean up temp files
    try:
        shutil.rmtree(temp_dir)
        print("\nTemporary test directory cleaned up.")
    except Exception as e:
        print(f"\nWarning: could not delete temporary test directory: {e}")

    print("\n=== INTEGRATION TEST PASSED SUCCESSFULLY! ===")


if __name__ == "__main__":
    # Windows multiprocessing protection
    import multiprocessing
    multiprocessing.freeze_support()
    run_verification()

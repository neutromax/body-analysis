# Automated Body Language Annotator (v3)

A research-grade Python desktop application that automatically generates a Body Language dataset from YouTube videos using MediaPipe Pose and geometry-based mathematics. No manual annotation, no machine learning model training, and no emotion recognition required.

## Key Features

- **Embedded YouTube Player**: Powered by `pywebview` (WebView2 on Windows) running in a child process to bypass UI thread conflicts.
- **MediaPipe Pose Integration**: Uses the modern Google MediaPipe Tasks API (`PoseLandmarker`) to detect pose landmarks at 1 Frame Per Second (FPS).
- **Geometry-Based Math Classifiers**: Classifies Hand Movement, Leg Movement, Head Direction, and Body Lean as `-1` (Left) / `0` (Center) / `1` (Right).
- **Statistical Calibration**: Calibrates motion thresholds dynamically from the first 3 seconds of the video using natural variance ($\mu \pm k\sigma$).
- **Aggregated Outputs**: Performs majority-vote aggregation over 10-second windows and appends rows directly to an existing Excel sheet.
- **Muted Autoplay Bypass**: Loads the video muted with `playsinline` to satisfy modern browser unmuted media block policies.
- **Robust Exception Handling**: Detects restricted/non-embeddable videos (e.g. YouTube error codes 101, 150, 153) and re-enables inputs cleanly without hanging.

## Quick Start

### 1. Install Dependencies

Ensure you have Python 3.10 installed on your system. Then, run:

```powershell
cd e:\projects\internship_iit
& "C:\Users\LENOVO\AppData\Local\Programs\Python\Python310\python.exe" -m pip install -r auto_annotator\requirements.txt
```

### 2. Launch the Application

Double-click the startup script:
- `run_auto_annotator.bat`

Or run directly from your terminal:
```powershell
& "C:\Users\LENOVO\AppData\Local\Programs\Python\Python310\python.exe" auto_annotator\main_app.py
```

*Note: On first startup, the app will download the MediaPipe pose landmarker model file (~13 MB) automatically.*

---

## Testing & Verification

### 1. Automated Math & Logic Unit Tests
To verify all coordinate geometry formulas, thresholding calibration, classifier mappings, and majority-vote tie-breaking:
```powershell
& "C:\Users\LENOVO\AppData\Local\Programs\Python\Python310\python.exe" -m pytest tests/test_auto_annotator.py -v
```

### 2. End-to-End Pipeline Integration Test
To run a complete programmatic simulation of the video player, frame capture, MediaPipe inference, calibration, and Excel writing:
```powershell
& "C:\Users\LENOVO\AppData\Local\Programs\Python\Python310\python.exe" verify_integration.py
```

---

## Project Structure

```
internship_iit/
├── auto_annotator/          # Core automated annotator system
│   ├── main_app.py          # Tkinter dark-themed control panel
│   ├── auto_config.py       # Configuration parameters and constants
│   ├── geometry.py          # Pure vector geometry calculations
│   ├── pose_engine.py       # MediaPipe Pose Landmarker Tasks API wrapper
│   ├── feature_classifier.py# Mappings from numbers to -1 / 0 / 1 labels
│   ├── calibration.py       # Natural variance baseline calibration
│   ├── aggregator.py        # Majority vote window aggregation
│   ├── excel_writer.py      # openpyxl Excel spreadsheet appender
│   ├── frame_capture.py     # mss desktop screen-region screenshot utility
│   ├── webview_player.py    # Subprocess-isolated pywebview YouTube player
│   └── requirements.txt     # Python requirements
├── tests/
│   ├── __init__.py
│   └── test_auto_annotator.py # 17/17 passing unit tests
├── verify_integration.py    # Headless E2E verification script
├── run_auto_annotator.bat   # App startup shortcut script
├── .gitignore               # Ignored local logs and outputs
└── README.md                # This documentation
```

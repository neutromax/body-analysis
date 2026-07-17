# Body Language Annotation System (v2)

A PyQt6 desktop application for manually annotating body language features (Hand, Leg, Head, Body movements) directly from YouTube videos using an embedded player. Designed for research workflows where videos are segmented into 10-second windows with 1-second observations.

## Features

- **Embedded YouTube Player**: Annotate directly while watching the online video inside the application via the YouTube IFrame Player API. No downloading required!
- **10-Second Windows**: Videos are automatically segmented into fixed annotation windows.
- **Per-Second Observations**: 10 observations per window (one observation every 1 second).
- **4 Body Features**: Hand Movement, Leg Movement, Head Bend, Body Lean.
- **Keyboard Shortcuts**: Rapid annotation with single keystrokes (Q/W/E, A/S/D, Z/X/C, 1/2/3).
- **Auto-Save**: Every observation is saved automatically to JSON. If the application closes unexpectedly, it resumes exactly where you stopped.
- **Majority Vote**: Computes the final window label using majority-vote logic. Ties default to Neutral (0).
- **Export**: Generates clean CSV and Excel files (Excel contains both aggregated data and raw audit logs).

## Quick Start

### 1. Install Dependencies

Ensure you have Python 3.8+ installed. Then run:

```bash
cd e:\projects\internship_iit
pip install -r requirements.txt
```

*Note: Since the video is streamed online, you do not need `ffmpeg` or `yt-dlp`.*

### 2. Launch the Application

```bash
python main.py
```

### 3. Annotating Workflow

1. Paste a YouTube URL on the Home Screen (e.g., `https://www.youtube.com/watch?v=dQw4w9WgXcQ`).
2. Select the video category and click **Load Video**.
3. Use the keyboard shortcuts to annotate the body features for the current observation:
   - **Hand**: `Q` (Left) | `W` (Neutral) | `E` (Right)
   - **Leg**: `A` (Left) | `S` (Neutral) | `D` (Right)
   - **Head**: `Z` (Left) | `X` (Neutral/Center) | `C` (Right)
   - **Body**: `1` (Left) | `2` (Neutral/Center) | `3` (Right)
4. Press **Space** to confirm the current observation. If any feature is left unannotated, it defaults to Neutral (0). The player will automatically seek to the next second and pause.
5. After 10 observations, the window advances, and the 10x4 mini-grid refreshes.
6. Use the **Arrow Keys** to step backward or forward through observations.
7. Use **Shift + Left/Right Arrow Keys** to jump to the previous or next window.

### 4. Exporting

Access **File** in the menu bar to export your annotations:
- **Export to CSV**: Generates a CSV containing only the time window and aggregated results.
- **Export to Excel**: Generates a spreadsheet with two tabs: one for the final aggregated dataset and one for raw per-second observations.

## Project Structure

```
internship_iit/
├── main.py                  # Entry point
├── config.py                # All constants, paths, shortcuts, and colors
├── requirements.txt         # Dependencies (PyQt6, pandas, openpyxl)
├── core/
│   ├── models.py            # Observation, Window, and Session dataclasses
│   ├── segmenter.py         # Window/observation timestamp mathematics
│   └── session_manager.py   # State tracking & workflow logic
├── gui/
│   ├── main_window.py       # Stacked view container & menu bar
│   ├── home_panel.py        # Home screen for URL entry and session resume
│   ├── player_widget.py     # Chromium QWebEngineView wrapper
│   ├── player_bridge.py     # Python-JS QWebChannel bridge
│   ├── annotation_panel.py  # Controls & buttons for annotations
│   ├── progress_panel.py    # Detailed progress & position displays
│   ├── observation_grid.py  # 10x4 mini status grid
│   └── styles.py            # Centralized dark-mode stylesheet (QSS)
├── storage/
│   └── json_store.py        # File persistence, backups, & session listings
├── aggregation/
│   └── majority_vote.py     # Majority voting logic & tie-breaking
├── export/
│   └── exporter.py          # CSV & Excel exporting
├── utils/
│   └── helpers.py           # YouTube URL parsing and time formatting
├── resources/
│   └── player.html          # Embedded HTML page running the IFrame API
├── data/                    # JSON sessions and exports (git-ignored)
└── tests/                   # Python unit tests (pytest)
```

## Running Tests

Verify the core logic, segmenter, and aggregation functions by running:

```bash
python -m pytest tests/ -v
```

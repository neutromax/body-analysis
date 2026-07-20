"""
Auto Annotator — Main Application
====================================

Tkinter GUI shell that wires everything together.

This is the entry point: it creates the control panel window, connects
button clicks to the session controller, and drives the in-memory
PyAV stream reader (no embedded browser needed).

Layout:
    ┌───────────────────────────────────────────────────────┐
    │  [Choose Excel]  [YouTube URL input]  [Load Video]    │
    │  [Start]  [Pause]  [Stop]                             │
    ├───────────────────────────────────────────────────────┤
    │  LIVE STATUS                                          │
    │  Window: 0:20–0:30  |  Time: 0:23                    │
    │  Calibration: ✓ Complete                              │
    │  Hand: → Right  |  Leg: ○ Neutral                    │
    │  Head: ← Left   |  Body: ○ Center                    │
    ├───────────────────────────────────────────────────────┤
    │  Progress: ████████░░  7/10  |  Rows: 12  |  Running │
    └───────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

# Ensure the auto_annotator package is on the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auto_config import (
    APP_TITLE,
    APP_WIDTH,
    APP_HEIGHT,
    COLORS,
    ENCODING,
    FEATURES,
    CALIBRATION_DURATION,
    OBSERVATIONS_PER_WINDOW,
)
from session_controller import SessionController, SessionState
from video_queue import VideoQueueManager, extract_video_id

logger = logging.getLogger(__name__)


class AutoAnnotatorApp:
    """
    The main Tkinter application — control panel + status dashboard.
    """

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry(f"{APP_WIDTH}x{APP_HEIGHT}")
        self.root.configure(bg=COLORS["bg"])
        self.root.resizable(True, True)
        self.root.minsize(460, 500)

        # State
        self._excel_path: Optional[str] = None
        self._queue_path: Optional[str] = None
        self._queue_mgr: Optional[VideoQueueManager] = None
        self._active_video_info: Optional[dict] = None
        self._resume_sec: int = 0
        self._initial_thresholds = None
        self._controller: Optional[SessionController] = None
        self._youtube_url: str = ""

        # Build the UI
        self._build_gui()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── GUI Construction ───────────────────────────────────────────────

    def _build_gui(self) -> None:
        """Assemble all GUI sections."""
        # Configure ttk styles for dark theme.
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background=COLORS["bg"])
        style.configure("Card.TFrame", background=COLORS["bg_light"])
        style.configure(
            "Dark.TLabel",
            background=COLORS["bg"],
            foreground=COLORS["text"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Header.TLabel",
            background=COLORS["bg"],
            foreground=COLORS["accent"],
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "Status.TLabel",
            background=COLORS["bg_light"],
            foreground=COLORS["text"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Value.TLabel",
            background=COLORS["bg_light"],
            foreground=COLORS["text"],
            font=("Segoe UI Semibold", 11),
        )
        style.configure(
            "Dark.TButton",
            background=COLORS["bg_card"],
            foreground=COLORS["text"],
            font=("Segoe UI", 10),
            padding=(12, 6),
        )
        style.map(
            "Dark.TButton",
            background=[("active", COLORS["accent"])],
        )
        style.configure(
            "Accent.TButton",
            background=COLORS["accent"],
            foreground="#ffffff",
            font=("Segoe UI Semibold", 10),
            padding=(14, 7),
        )
        style.map(
            "Accent.TButton",
            background=[("active", COLORS["accent_hover"])],
        )
        style.configure(
            "Dark.TEntry",
            fieldbackground=COLORS["bg_card"],
            foreground=COLORS["text"],
            insertcolor=COLORS["text"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "green.Horizontal.TProgressbar",
            troughcolor=COLORS["bg_card"],
            background=COLORS["success"],
        )

        main_frame = ttk.Frame(self.root, style="Dark.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        self._build_input_section(main_frame)
        self._build_controls_section(main_frame)
        self._build_status_section(main_frame)
        self._build_progress_section(main_frame)

    def _build_input_section(self, parent: ttk.Frame) -> None:
        """Excel files chooser + YouTube URL input + Load Video button."""
        section = ttk.Frame(parent, style="Dark.TFrame")
        section.pack(fill=tk.X, pady=(0, 10))

        # Row 1: Output Excel file
        row1 = ttk.Frame(section, style="Dark.TFrame")
        row1.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(row1, text="Output Excel:", style="Dark.TLabel").pack(side=tk.LEFT)
        self._excel_label = ttk.Label(
            row1, text="No file selected", style="Dark.TLabel",
        )
        self._excel_label.pack(side=tk.LEFT, padx=(8, 0), expand=True, fill=tk.X)
        ttk.Button(
            row1, text="Choose Output .xlsx", style="Dark.TButton",
            command=self._choose_excel,
        ).pack(side=tk.RIGHT)

        # Row 1b: Link Queue File
        row1b = ttk.Frame(section, style="Dark.TFrame")
        row1b.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(row1b, text="Link Queue:", style="Dark.TLabel").pack(side=tk.LEFT)
        self._queue_label = ttk.Label(
            row1b, text="No file selected", style="Dark.TLabel",
        )
        self._queue_label.pack(side=tk.LEFT, padx=(8, 0), expand=True, fill=tk.X)
        ttk.Button(
            row1b, text="Choose Queue .xlsx", style="Dark.TButton",
            command=self._choose_queue,
        ).pack(side=tk.RIGHT)

        # Row 2: YouTube URL
        row2 = ttk.Frame(section, style="Dark.TFrame")
        row2.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(row2, text="YouTube URL:", style="Dark.TLabel").pack(side=tk.LEFT)
        self._url_var = tk.StringVar()
        self._url_entry = ttk.Entry(
            row2, textvariable=self._url_var, style="Dark.TEntry",
        )
        self._url_entry.pack(side=tk.LEFT, padx=(8, 8), expand=True, fill=tk.X)
        self._load_btn = ttk.Button(
            row2, text="Load Video", style="Accent.TButton",
            command=self._load_video,
        )
        self._load_btn.pack(side=tk.RIGHT)

        # Row 3: Resume Info / Queue status
        row3 = ttk.Frame(section, style="Dark.TFrame")
        row3.pack(fill=tk.X, pady=(0, 6))
        self._resume_status_var = tk.StringVar(value="Queue: No queue loaded")
        ttk.Label(
            row3, textvariable=self._resume_status_var, style="Dark.TLabel",
            foreground=COLORS["warning"],
        ).pack(side=tk.LEFT)

    def _build_controls_section(self, parent: ttk.Frame) -> None:
        """Start / Pause / Stop / Mark Done buttons."""
        section = ttk.Frame(parent, style="Dark.TFrame")
        section.pack(fill=tk.X, pady=(0, 14))

        self._start_btn = ttk.Button(
            section, text="▶  Start", style="Accent.TButton",
            command=self._start, state=tk.DISABLED,
        )
        self._start_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._pause_btn = ttk.Button(
            section, text="⏸  Pause", style="Dark.TButton",
            command=self._pause, state=tk.DISABLED,
        )
        self._pause_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._stop_btn = ttk.Button(
            section, text="⏹  Stop", style="Dark.TButton",
            command=self._stop, state=tk.DISABLED,
        )
        self._stop_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._skip_btn = ttk.Button(
            section, text="⏭  Skip/Done", style="Dark.TButton",
            command=self._skip_done, state=tk.DISABLED,
        )
        self._skip_btn.pack(side=tk.LEFT)

    def _build_status_section(self, parent: ttk.Frame) -> None:
        """Live status display — current window, features, calibration."""
        section = ttk.LabelFrame(
            parent, text="  Live Status  ",
            style="Card.TFrame",
            labelanchor="n",
        )
        # Configure the label of the LabelFrame
        style = ttk.Style()
        style.configure(
            "Card.TLabelframe",
            background=COLORS["bg_light"],
            foreground=COLORS["accent"],
            font=("Segoe UI Semibold", 11),
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=COLORS["bg"],
            foreground=COLORS["accent"],
            font=("Segoe UI Semibold", 11),
        )
        section = ttk.LabelFrame(
            parent, text="  Live Status  ",
            style="Card.TLabelframe",
        )
        section.pack(fill=tk.BOTH, expand=True, pady=(0, 14))

        inner = ttk.Frame(section, style="Card.TFrame")
        inner.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        # Grid layout for status fields.
        labels = [
            ("Current Window:", "window"),
            ("Current Time:", "time"),
            ("Calibration:", "calibration"),
            ("", ""),  # separator
            ("✋ Hand:", "hand"),
            ("🦵 Leg:", "leg"),
            ("🗣️ Head:", "head"),
            ("🧍 Body:", "body"),
        ]

        self._status_vars: dict[str, tk.StringVar] = {}

        for row_idx, (label_text, key) in enumerate(labels):
            if not key:
                # Separator
                sep = ttk.Separator(inner, orient=tk.HORIZONTAL)
                sep.grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=6)
                continue

            ttk.Label(
                inner, text=label_text, style="Status.TLabel",
            ).grid(row=row_idx, column=0, sticky=tk.W, padx=(0, 16), pady=3)

            var = tk.StringVar(value="—")
            self._status_vars[key] = var
            ttk.Label(
                inner, textvariable=var, style="Value.TLabel",
            ).grid(row=row_idx, column=1, sticky=tk.W, pady=3)

        inner.columnconfigure(1, weight=1)

    def _build_progress_section(self, parent: ttk.Frame) -> None:
        """Progress bar + rows written + session status."""
        section = ttk.Frame(parent, style="Dark.TFrame")
        section.pack(fill=tk.X, pady=(0, 4))

        # Progress bar (readings within current window)
        self._progress_var = tk.IntVar(value=0)
        self._progress_bar = ttk.Progressbar(
            section,
            orient=tk.HORIZONTAL,
            mode="determinate",
            maximum=OBSERVATIONS_PER_WINDOW,
            variable=self._progress_var,
            style="green.Horizontal.TProgressbar",
        )
        self._progress_bar.pack(fill=tk.X, pady=(0, 6))

        # Bottom status line
        bottom = ttk.Frame(section, style="Dark.TFrame")
        bottom.pack(fill=tk.X)

        self._rows_var = tk.StringVar(value="Rows: 0")
        ttk.Label(bottom, textvariable=self._rows_var, style="Dark.TLabel").pack(
            side=tk.LEFT
        )

        self._session_status_var = tk.StringVar(value="Idle")
        ttk.Label(
            bottom, textvariable=self._session_status_var, style="Dark.TLabel",
        ).pack(side=tk.LEFT, padx=(20, 0))

        self._pose_var = tk.StringVar(value="")
        ttk.Label(bottom, textvariable=self._pose_var, style="Dark.TLabel").pack(
            side=tk.RIGHT
        )

    # ── Button handlers ────────────────────────────────────────────────

    def _choose_excel(self) -> None:
        """Open file dialog to select an existing .xlsx file."""
        path = filedialog.askopenfilename(
            title="Select Output Excel File",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
        )
        if path:
            self._excel_path = path
            basename = os.path.basename(path)
            self._excel_label.configure(text=basename)
            logger.info("Output Excel file selected: %s", path)
            if self._youtube_url or self._active_video_info:
                self._start_btn.configure(state=tk.NORMAL)

    def _choose_queue(self) -> None:
        """Open file dialog to select the Link Queue Excel file."""
        path = filedialog.askopenfilename(
            title="Select Link Queue Excel File",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
        )
        if path:
            try:
                self._queue_path = path
                self._queue_mgr = VideoQueueManager(path)
                basename = os.path.basename(path)
                self._queue_label.configure(text=basename)
                logger.info("Link Queue Excel file selected: %s", path)
                self._update_next_video_from_queue()
            except Exception as e:
                messagebox.showerror("Error Opening Queue", str(e))
                self._queue_path = None
                self._queue_mgr = None
                self._queue_label.configure(text="No file selected")
                self._resume_status_var.set("Queue: Error loading queue file")

    def _update_next_video_from_queue(self) -> None:
        """Fetch the next video from queue, run duplicate checks and cross-check checkpoints."""
        if not self._queue_mgr:
            return

        try:
            next_vid, warnings = self._queue_mgr.get_next_video()
            if warnings:
                messagebox.showwarning("Queue Warning", "\n".join(warnings))

            if next_vid:
                self._active_video_info = next_vid
                video_id = next_vid["video_id"]
                video_link = next_vid["video_link"]
                last_sec = next_vid["last_sec_completed"]
                
                # Cross-check checkpoint
                is_ok, msg, checkpoint_data = self._queue_mgr.cross_check_checkpoint(video_id, last_sec)
                if not is_ok and msg:
                    messagebox.showwarning("Checkpoint Disagreement", msg)
                
                # Load thresholds from checkpoint if available
                self._initial_thresholds = None
                if checkpoint_data:
                    t_data = checkpoint_data.get("session_info", {}).get("thresholds")
                    if t_data:
                        from feature_classifier import Thresholds
                        self._initial_thresholds = Thresholds(
                            T_body=t_data.get("T_body"),
                            T_head=t_data.get("T_head"),
                            T_hand=t_data.get("T_hand"),
                            T_leg=t_data.get("T_leg"),
                            calibrated=True
                        )
                
                # Populate URL entry
                self._url_var.set(video_link)
                self._youtube_url = video_link
                
                # Update resume label
                if last_sec is not None and last_sec >= 0:
                    self._resume_sec = last_sec + 1
                    self._resume_status_var.set(
                        f"Queue: video {video_id} resumes at sec {self._resume_sec}."
                    )
                else:
                    self._resume_sec = 0
                    self._resume_status_var.set(
                        f"Queue: video {video_id} starts at sec 0."
                    )

                # Update buttons
                self._load_btn.configure(state=tk.DISABLED)
                if self._excel_path:
                    self._start_btn.configure(state=tk.NORMAL)
                    self._session_status_var.set("Ready — press Start")
            else:
                self._active_video_info = None
                self._resume_sec = 0
                self._initial_thresholds = None
                self._url_var.set("")
                self._youtube_url = ""
                self._resume_status_var.set("Queue complete — nothing left to process")
                self._load_btn.configure(state=tk.NORMAL)
                self._start_btn.configure(state=tk.DISABLED)
                self._session_status_var.set("Queue Complete")
        except Exception as e:
            logger.error("Failed to update next video from queue: %s", e)
            messagebox.showerror("Queue Error", f"Error updating next video from queue: {e}")

    def _load_video(self) -> None:
        """Validate the YouTube URL and enable the Start button."""
        url = self._url_var.get().strip()
        if not url:
            messagebox.showwarning("No URL", "Please paste a YouTube URL.")
            return

        if not self._excel_path:
            messagebox.showwarning("No Excel File", "Please select an Excel file first.")
            return

        # Basic URL validation.
        if "youtube.com" not in url and "youtu.be" not in url:
            messagebox.showwarning("Invalid URL", "Please enter a valid YouTube URL.")
            return

        # Check duplicate-prevention rule if queue is loaded
        try:
            vid_id = extract_video_id(url)
            if self._queue_mgr and self._queue_mgr.is_video_done(vid_id):
                messagebox.showerror(
                    "Duplicate Prevention",
                    f"Video {vid_id} is already marked Done in the queue and cannot be reprocessed."
                )
                return
        except Exception as e:
            messagebox.showwarning("Invalid URL", f"Could not extract Video ID from URL: {e}")
            return

        self._youtube_url = url
        self._load_btn.configure(state=tk.DISABLED)
        self._session_status_var.set("Ready — press Start")
        self._start_btn.configure(state=tk.NORMAL)
        logger.info("URL validated: %s", url)

    def _start(self) -> None:
        """Start the annotation session."""
        if not self._excel_path or not self._youtube_url:
            return

        # Create a new controller for this session.
        self._controller = SessionController(
            excel_path=self._excel_path,
            video_url=self._youtube_url,
            root_after=self.root.after,
            gui_callback=self._update_gui,
            resume_sec=self._resume_sec,
            initial_thresholds=self._initial_thresholds,
            queue_mgr=self._queue_mgr,
        )
        self._session_status_var.set("Resolving stream URL...")
        self._controller.start()

        # Update button states.
        self._start_btn.configure(state=tk.DISABLED)
        self._pause_btn.configure(state=tk.NORMAL)
        self._stop_btn.configure(state=tk.NORMAL)
        self._skip_btn.configure(state=tk.NORMAL)
        self._load_btn.configure(state=tk.DISABLED)

    def _pause(self) -> None:
        """Toggle pause/resume."""
        if not self._controller:
            return

        if self._controller.state == SessionState.PAUSED:
            self._controller.resume()
            self._pause_btn.configure(text="⏸  Pause")
        else:
            self._controller.pause()
            self._pause_btn.configure(text="▶  Resume")

    def _stop(self) -> None:
        """Stop the annotation session."""
        if self._controller:
            self._controller.stop()
            self._controller = None

        self._start_btn.configure(state=tk.NORMAL)
        self._pause_btn.configure(state=tk.DISABLED, text="⏸  Pause")
        self._stop_btn.configure(state=tk.DISABLED)
        self._skip_btn.configure(state=tk.DISABLED)
        self._load_btn.configure(state=tk.NORMAL)
        self._session_status_var.set("Stopped")

    def _skip_done(self) -> None:
        """Explicitly mark the current video as Finished/Done and move to next."""
        if not self._controller:
            return
        
        # Stop session (writes any partial window data)
        self._controller.stop()
        
        # Mark as done in the queue manager
        if self._queue_mgr and self._active_video_info:
            try:
                self._queue_mgr.mark_done(self._active_video_info["video_id"])
            except Exception as e:
                logger.error("Failed to mark done in queue sheet: %s", e)
            
        self._handle_video_ended()

    def _on_video_ended(self) -> None:
        """Called when the YouTube video finishes playing."""
        # This callback comes from the pywebview thread, so schedule
        # GUI updates on the main thread.
        self.root.after(0, self._handle_video_ended)

    def _handle_video_ended(self) -> None:
        """Handle video end on the main thread."""
        self._session_status_var.set("✅ Video Finished")
        
        if self._controller:
            try:
                self._controller.stop()
            except Exception:
                pass
            self._controller = None

        self._start_btn.configure(state=tk.NORMAL)
        self._pause_btn.configure(state=tk.DISABLED, text="⏸  Pause")
        self._stop_btn.configure(state=tk.DISABLED)
        self._skip_btn.configure(state=tk.DISABLED)
        self._load_btn.configure(state=tk.NORMAL)

        # Auto-advance to next video if in queue mode
        if self._queue_mgr:
            self._update_next_video_from_queue()
            if self._active_video_info:
                logger.info("Auto-advancing to next video in 1 second...")
                self.root.after(1000, self._start)
            else:
                messagebox.showinfo("Queue Complete", "Queue complete — nothing left to process.")

    # ── GUI updates ────────────────────────────────────────────────────

    def _update_gui(self, status: dict) -> None:
        """
        Called by session_controller every second with current status.

        Updates all status labels, progress bar, and bottom info line.
        """
        # State
        state = status.get("state", "UNKNOWN")
        self._session_status_var.set(state.replace("_", " ").title())

        # Window info
        self._status_vars["window"].set(status.get("time_window", "—"))

        # Current time
        current = status.get("current_time", 0)
        mins, secs = divmod(int(current), 60)
        duration = status.get("video_duration", 0)
        dur_m, dur_s = divmod(int(duration), 60)
        self._status_vars["time"].set(
            f"{mins}:{secs:02d} / {dur_m}:{dur_s:02d}"
        )

        # Calibration
        if state == "CALIBRATING":
            cal_prog = status.get("calibration_progress", 0)
            cal_total = status.get("calibration_total", CALIBRATION_DURATION)
            self._status_vars["calibration"].set(
                f"⏳ {cal_prog}/{cal_total} seconds..."
            )
        elif status.get("thresholds"):
            t = status["thresholds"]
            cal_text = f"✓ T_body={t.T_body:.1f}°  T_head={t.T_head:.3f}"
            if t.calibrated:
                cal_text = "✓ " + cal_text
            else:
                cal_text = "⚠ Fallback: " + cal_text
            self._status_vars["calibration"].set(cal_text)
        else:
            self._status_vars["calibration"].set("—")

        # Feature readings
        reading = status.get("latest_reading", {})
        for feature in FEATURES:
            val = reading.get(feature)
            if val is None:
                display = "— (no data)"
            elif val == -1:
                display = "← Left (-1)"
            elif val == 1:
                display = "→ Right (1)"
            else:
                display = "○ Center (0)"
            self._status_vars[feature].set(display)

        # Progress bar
        readings_count = status.get("readings_in_window", 0)
        self._progress_var.set(readings_count)

        # Bottom info
        self._rows_var.set(f"Rows: {status.get('total_rows', 0)}")
        self._pose_var.set(
            "Pose: ✓" if status.get("pose_detected") else "Pose: ✗"
        )

        # Finished
        if status.get("finished"):
            self._handle_video_ended()

    # ── Cleanup ────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        """Handle window close — stop session, exit."""
        if self._controller and self._controller.state in (
            SessionState.CALIBRATING,
            SessionState.RUNNING,
            SessionState.PAUSED,
        ):
            self._controller.stop()

        self.root.destroy()

    def run(self) -> None:
        """Start the Tkinter event loop."""
        self.root.mainloop()


# ── Entry point ────────────────────────────────────────────────────────

def main() -> None:
    """Application entry point."""
    import multiprocessing
    multiprocessing.freeze_support()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Starting %s", APP_TITLE)

    app = AutoAnnotatorApp()
    app.run()


if __name__ == "__main__":
    main()

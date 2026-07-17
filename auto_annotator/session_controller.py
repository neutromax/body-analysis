"""
Auto Annotator — Session Controller
======================================

The per-second / per-10-second timing loop that orchestrates the entire
annotation pipeline.

Lifecycle:
    1. User clicks Start → controller starts calibration (K seconds).
    2. After calibration → enters main loop (1 capture/second).
    3. Each second: capture → pose detect → compute metrics → classify → store.
    4. Every 10 seconds: aggregate → write Excel row → update GUI.
    5. On Pause: suspends the timer loop; resumes on un-pause.
    6. On Stop or video end: writes any partial window, stops.

Threading model:
    The timing loop runs via Tkinter's root.after() mechanism, which
    keeps everything on the main thread and avoids race conditions with
    the GUI.  The only background thread is pywebview's own event loop.
"""

from __future__ import annotations

import logging
import time
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

from auto_config import (
    OBSERVATION_INTERVAL,
    OBSERVATIONS_PER_WINDOW,
    CALIBRATION_DURATION,
    ENCODING,
)
from frame_capture import capture_region
from pose_engine import PoseEngine
from feature_classifier import (
    FrameMetrics,
    FrameReading,
    Thresholds,
    compute_frame_metrics,
    classify_frame,
)
from calibration import Calibrator
from aggregator import aggregate_window
from excel_writer import ExcelWriter
from webview_player import WebViewPlayer

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """States of the annotation session."""
    IDLE = auto()
    CALIBRATING = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()
    FINISHED = auto()


def _format_time_window(window_index: int, window_duration: int) -> str:
    """
    Format a time window index as a human-readable range.

    Example: window_index=2, duration=10 → "0:20 – 0:30"
    """
    start = window_index * window_duration
    end = start + window_duration

    def _fmt(seconds: int) -> str:
        m, s = divmod(seconds, 60)
        return f"{m}:{s:02d}"

    return f"{_fmt(start)} – {_fmt(end)}"


class SessionController:
    """
    Orchestrates the capture → classify → aggregate → write pipeline.

    This is the "brain" of the application, connecting all modules.

    Attributes:
        state:           Current session state.
        _player:         WebView player instance.
        _engine:         MediaPipe Pose engine.
        _writer:         Excel writer instance.
        _calibrator:     Calibration data collector.
        _thresholds:     Per-video classification thresholds.
        _window_readings: Current window's per-second readings.
        _window_index:   Current 10-second window number (0-based).
        _prev_metrics:   Previous second's FrameMetrics (for velocity).
        _gui_callback:   Function to call with status updates for the GUI.
    """

    def __init__(
        self,
        player: WebViewPlayer,
        excel_path: str,
        video_url: str,
        root_after: Callable,
        gui_callback: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            player:       The WebView player (already loaded).
            excel_path:   Path to the .xlsx file to append to.
            video_url:    The YouTube URL being annotated.
            root_after:   Tkinter's root.after(ms, callback) for scheduling.
            gui_callback: Called with a status dict every second for GUI updates.
        """
        self._player = player
        self._root_after = root_after
        self._gui_callback = gui_callback
        self._video_url = video_url

        # Modules
        self._engine = PoseEngine()
        self._writer = ExcelWriter(excel_path)
        self._writer.start_video(video_url)
        self._calibrator = Calibrator()

        # State
        self.state = SessionState.IDLE
        self._thresholds: Optional[Thresholds] = None
        self._window_readings: List[FrameReading] = []
        self._window_index: int = 0
        self._prev_metrics: Optional[FrameMetrics] = None
        self._after_id: Optional[str] = None
        self._total_rows: int = 0

    # ── Public control methods ─────────────────────────────────────────

    def start(self) -> None:
        """Begin the annotation session (starts with calibration)."""
        if self.state not in (SessionState.IDLE, SessionState.STOPPED):
            logger.warning("Cannot start: state is %s", self.state)
            return

        logger.info("Session starting — calibration phase (%ds)", CALIBRATION_DURATION)
        self.state = SessionState.CALIBRATING
        self._prev_metrics = None
        self._window_readings = []
        self._window_index = 0
        self._calibrator = Calibrator()

        self._schedule_next_tick()

    def pause(self) -> None:
        """Pause the session (can be resumed)."""
        if self.state in (SessionState.CALIBRATING, SessionState.RUNNING):
            self.state = SessionState.PAUSED
            self._cancel_next_tick()
            logger.info("Session paused.")

    def resume(self) -> None:
        """Resume a paused session."""
        if self.state == SessionState.PAUSED:
            self.state = SessionState.RUNNING
            self._schedule_next_tick()
            logger.info("Session resumed.")

    def stop(self) -> None:
        """Stop the session. Writes any partial window data."""
        self._cancel_next_tick()
        self._flush_partial_window()
        self.state = SessionState.STOPPED
        self._engine.close()
        logger.info("Session stopped. Total rows: %d", self._total_rows)

    # ── Timing loop ────────────────────────────────────────────────────

    def _schedule_next_tick(self) -> None:
        """Schedule the next 1-second tick via Tkinter's event loop."""
        self._after_id = self._root_after(
            OBSERVATION_INTERVAL * 1000,
            self._tick,
        )

    def _cancel_next_tick(self) -> None:
        """Cancel any pending tick."""
        # We can't cancel root.after directly without root reference,
        # so we use the state check in _tick() as a guard instead.
        self._after_id = None

    def _tick(self) -> None:
        """
        One iteration of the per-second loop.

        This is called by Tkinter's after() mechanism, so it runs on the
        main thread.
        """
        # Guard: don't run if we've been paused/stopped since scheduling.
        if self.state not in (SessionState.CALIBRATING, SessionState.RUNNING):
            return

        # Check if video ended.
        if self._player.is_ended():
            logger.info("Video ended.")
            self._flush_partial_window()
            self.state = SessionState.FINISHED
            self._notify_gui(finished=True)
            self._engine.close()
            return

        # Check if video is actually playing.
        if not self._player.is_playing():
            # Video might be buffering or paused by user in the player.
            # Reschedule and try again.
            self._schedule_next_tick()
            return

        # ── Capture & detect ───────────────────────────────────────────
        reading, metrics = self._capture_and_classify()

        # ── Route to calibration or main loop ──────────────────────────
        if self.state == SessionState.CALIBRATING:
            self._handle_calibration_tick(metrics)
        elif self.state == SessionState.RUNNING:
            self._handle_running_tick(reading)

        # ── Update GUI ─────────────────────────────────────────────────
        self._notify_gui()

        # ── Schedule next tick ─────────────────────────────────────────
        if self.state in (SessionState.CALIBRATING, SessionState.RUNNING):
            self._schedule_next_tick()

    def _capture_and_classify(self) -> tuple[FrameReading, FrameMetrics]:
        """
        Capture one frame, detect pose, compute metrics, classify.

        Returns:
            (reading, metrics) — the classified reading and raw metrics.
            If capture or detection fails, returns an all-NA reading and
            empty metrics.
        """
        reading: FrameReading = {f: None for f in ["hand", "leg", "head", "body"]}
        metrics = FrameMetrics()

        try:
            # Get player bounding box for screen capture.
            bbox = self._player.get_player_bbox()
            if bbox is None:
                logger.warning("Cannot get player bbox — skipping frame.")
                return reading, metrics

            # Capture the player region.
            image = capture_region(bbox)

            # Run pose detection.
            landmarks = self._engine.detect(image)
            if landmarks is None:
                logger.debug("No pose detected — marking all features as NA.")
                self._prev_metrics = None
                return reading, metrics

            # Compute raw metrics.
            metrics = compute_frame_metrics(landmarks, self._prev_metrics)
            self._prev_metrics = metrics

            # Classify (only if we have thresholds — i.e. post-calibration).
            if self._thresholds is not None:
                reading = classify_frame(metrics, self._thresholds)

        except Exception as e:
            logger.error("Error during capture/classify: %s", e, exc_info=True)

        return reading, metrics

    # ── Calibration phase ──────────────────────────────────────────────

    def _handle_calibration_tick(self, metrics: FrameMetrics) -> None:
        """Process one calibration second."""
        self._calibrator.add_reading(metrics)

        if self._calibrator.is_complete:
            self._thresholds = self._calibrator.compute_thresholds()
            self.state = SessionState.RUNNING
            logger.info("Calibration complete → entering main loop.")

    # ── Main annotation loop ───────────────────────────────────────────

    def _handle_running_tick(self, reading: FrameReading) -> None:
        """Process one annotation second."""
        self._window_readings.append(reading)

        # Check if the window is full (10 readings).
        if len(self._window_readings) >= OBSERVATIONS_PER_WINDOW:
            self._complete_window()

    def _complete_window(self) -> None:
        """Aggregate and write one completed 10-second window."""
        result = aggregate_window(self._window_readings)
        time_window = _format_time_window(self._window_index, 10)

        try:
            self._total_rows = self._writer.append_row(time_window, result)
        except Exception as e:
            logger.error("Failed to write Excel row: %s", e, exc_info=True)

        logger.info(
            "Window %d (%s) → hand=%s, leg=%s, head=%s, body=%s",
            self._window_index, time_window,
            result.get("hand"), result.get("leg"),
            result.get("head"), result.get("body"),
        )

        # Reset for next window.
        self._window_readings = []
        self._window_index += 1

    def _flush_partial_window(self) -> None:
        """Write a partial window (< 10 readings) if any data exists."""
        if self._window_readings:
            logger.info(
                "Flushing partial window %d (%d readings)",
                self._window_index, len(self._window_readings),
            )
            self._complete_window()

    # ── GUI notification ───────────────────────────────────────────────

    def _notify_gui(self, finished: bool = False) -> None:
        """Send a status update to the GUI callback."""
        if self._gui_callback is None:
            return

        # Get the latest reading (last in the current window, if any).
        latest = self._window_readings[-1] if self._window_readings else {}

        status = {
            "state": self.state.name,
            "window_index": self._window_index,
            "time_window": _format_time_window(self._window_index, 10),
            "readings_in_window": len(self._window_readings),
            "total_rows": self._total_rows,
            "current_time": self._player.get_current_time(),
            "video_duration": self._player.get_duration(),
            "latest_reading": latest,
            "thresholds": self._thresholds,
            "calibration_progress": self._calibrator.readings_collected,
            "calibration_total": CALIBRATION_DURATION,
            "finished": finished,
            "pose_detected": bool(latest),
        }

        try:
            self._gui_callback(status)
        except Exception as e:
            logger.error("GUI callback error: %s", e)

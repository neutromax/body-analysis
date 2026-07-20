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
import os
import time
import math
import json
from pathlib import Path
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from auto_config import (
    OBSERVATION_INTERVAL,
    OBSERVATIONS_PER_WINDOW,
    CALIBRATION_DURATION,
    ENCODING,
)
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
from stream_reader import StreamReader
from video_queue import extract_video_id

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
    Format a time window index as a raw seconds range.

    Example: window_index=2, duration=10 → "20-30"
    """
    start = window_index * window_duration
    end = start + window_duration
    return f"{start}-{end}"


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
        excel_path: str,
        video_url: str,
        root_after: Callable,
        gui_callback: Optional[Callable] = None,
        resume_sec: int = 0,
        initial_thresholds: Optional[Thresholds] = None,
        queue_mgr: Optional[Any] = None,
    ) -> None:
        """
        Args:
            excel_path:   Path to the .xlsx file to append to.
            video_url:    The YouTube URL being annotated.
            root_after:   Tkinter's root.after(ms, callback) for scheduling.
            gui_callback: Called with a status dict every second for GUI updates.
            resume_sec:   Second to resume from (0 if start from beginning).
            initial_thresholds: Loaded thresholds if resuming, otherwise None.
            queue_mgr:    The VideoQueueManager instance (if running in queue mode).
        """
        self._root_after = root_after
        self._gui_callback = gui_callback
        self._video_url = video_url
        self._resume_sec = resume_sec
        self._queue_mgr = queue_mgr

        # Extract video ID & path for checkpoints
        self._video_id = extract_video_id(video_url)
        self._checkpoint_path = Path(__file__).parent.parent / "data" / "sessions" / f"{self._video_id}.json"

        # Modules
        self._engine = PoseEngine()
        self._writer = ExcelWriter(excel_path)
        self._writer.start_video(video_url)
        self._calibrator = Calibrator()

        # In-memory stream reader (PyAV + yt-dlp)
        self._stream_reader = StreamReader(youtube_url=video_url, seek_time=float(resume_sec))
        self._session_start_time: float = 0.0

        # State
        self.state = SessionState.IDLE
        self._thresholds: Optional[Thresholds] = initial_thresholds
        self._window_readings: List[FrameReading] = []
        self._window_index: int = resume_sec // 10
        self._calibration_metrics: List[FrameMetrics] = []
        self._prev_metrics: Optional[FrameMetrics] = None
        self._after_id: Optional[str] = None
        self._total_rows: int = 0
        self._debug_frame_count: int = 0
        self._last_processed_pts: Optional[float] = None
        self._last_pose_pts: Optional[float] = None

    # ── Public control methods ─────────────────────────────────────────

    def start(self) -> None:
        """Begin the annotation session (starts with calibration)."""
        if self.state not in (SessionState.IDLE, SessionState.STOPPED):
            logger.warning("Cannot start: state is %s", self.state)
            return

        self._prev_metrics = None
        self._window_readings = []
        
        if self._resume_sec > 0:
            logger.info("Session starting — resuming from second %d (window %d)", self._resume_sec, self._window_index)
            self.state = SessionState.RUNNING
            self._session_start_time = time.time() - self._resume_sec
        else:
            logger.info("Session starting — calibration phase (%ds)", CALIBRATION_DURATION)
            self.state = SessionState.CALIBRATING
            self._window_index = 0
            self._calibrator = Calibrator()
            self._session_start_time = time.time()

        # Start in-memory stream decoding
        self._stream_reader.start()
        logger.info("In-memory stream reader started.")

        # Initialize OpenCV visualization window resizable
        try:
            cv2.namedWindow("Auto Annotator — Pose Tracking", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Auto Annotator — Pose Tracking", 854, 480)
        except Exception as e:
            logger.warning("Could not initialize OpenCV visualization window: %s", e)

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
        self._stream_reader.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
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

        # Check if the stream has finished decoding all frames.
        if self._stream_reader.is_finished:
            logger.info("Video stream finished decoding.")
            self._flush_partial_window()
            if self._queue_mgr:
                if self._stream_reader.is_genuine_end:
                    self._queue_mgr.mark_done(self._video_id)
                else:
                    # Unexpected stream termination
                    resolve_err = getattr(self._stream_reader, "_resolve_error", None)
                    if self.state in (SessionState.CALIBRATING, SessionState.RUNNING):
                        if resolve_err or self._total_rows == 0:
                            reason = resolve_err or "Stream disconnected prematurely."
                            self._queue_mgr.mark_error(self._video_id, reason)
                        else:
                            last_sec = self._window_index * 10 + len(self._window_readings) - 1
                            if last_sec >= 0:
                                self._queue_mgr.mark_progress(self._video_id, last_sec)
            self.state = SessionState.FINISHED
            self._notify_gui(finished=True)
            self._engine.close()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            return

        # Skip this tick if no frame is available yet (stream still
        # resolving the URL or buffering the first frame).
        if self._stream_reader.current_frame is None:
            logger.info("No frame available yet — waiting for stream to start...")
            self._notify_gui()
            self._schedule_next_tick()
            return

        # Skip if this frame has already been processed (prevents duplicates during buffering/lags)
        stream_pts = self._stream_reader.current_pts
        if self._last_processed_pts is not None and abs(stream_pts - self._last_processed_pts) < 0.1:
            self._after_id = self._root_after(100, self._tick)
            return

        # ── Capture & detect ───────────────────────────────────────────
        reading, metrics = self._capture_and_classify()
        self._last_processed_pts = stream_pts

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
            # Get the current in-memory decoded frame from the stream reader.
            image = self._stream_reader.current_frame
            if image is None:
                logger.warning("No decoded frame available yet — skipping.")
                return reading, metrics

            # Save first few frames to disk for diagnostic inspection.
            if self._debug_frame_count < 3:
                self._save_debug_frame(image)

            # Run pose detection (VIDEO mode with temporal tracking).
            pts_ms = int(self._stream_reader.current_pts * 1000)
            landmarks = self._engine.detect(image, timestamp_ms=pts_ms)
            if landmarks is None:
                logger.info(
                    "No pose detected — shape=%s, dtype=%s, mean_px=%.1f, PTS=%.1fs",
                    image.shape, image.dtype, float(image.mean()),
                    self._stream_reader.current_pts,
                )
                # Keep prev_metrics so velocity can be computed when
                # detection resumes (don't break the chain).
                self._visualize_frame(image, None, reading)
                return reading, metrics

            # Compute raw metrics. Only compute velocity if consecutive frames are close in time (<= 1.5s).
            current_pts = self._stream_reader.current_pts
            if self._last_pose_pts is not None and (current_pts - self._last_pose_pts) <= 1.5:
                metrics = compute_frame_metrics(landmarks, self._prev_metrics)
            else:
                metrics = compute_frame_metrics(landmarks, None)

            self._prev_metrics = metrics
            self._last_pose_pts = current_pts

            logger.info(
                "Pose DETECTED — PTS=%.1fs, angle=%.1f°, head_x=%s",
                current_pts,
                metrics.body_angle_deg if metrics.body_angle_deg is not None else 0,
                metrics.head_local_x,
            )

            # Classify (only if we have thresholds — i.e. post-calibration).
            if self._thresholds is not None:
                reading = classify_frame(metrics, self._thresholds)

            self._visualize_frame(image, landmarks, reading)

        except Exception as e:
            logger.error("Error during capture/classify: %s", e, exc_info=True)

        return reading, metrics

    def _visualize_frame(
        self,
        image: np.ndarray,
        landmarks: Optional[dict],
        reading: FrameReading,
    ) -> None:
        """Draw pose landmarks, skeleton, and HUD of classified states on the frame."""
        try:
            visual_image = image.copy()
            H, W, _ = visual_image.shape

            # Extract/estimate landmarks for drawing
            draw_landmarks = landmarks.copy() if landmarks is not None else None
            
            if draw_landmarks is not None:
                # Check for virtual hips: if shoulders are visible but hips are not
                has_hips = (
                    "LEFT_HIP" in draw_landmarks and draw_landmarks["LEFT_HIP"][2] >= 0.3 and
                    "RIGHT_HIP" in draw_landmarks and draw_landmarks["RIGHT_HIP"][2] >= 0.3
                )
                has_shoulders = (
                    "LEFT_SHOULDER" in draw_landmarks and draw_landmarks["LEFT_SHOULDER"][2] >= 0.3 and
                    "RIGHT_SHOULDER" in draw_landmarks and draw_landmarks["RIGHT_SHOULDER"][2] >= 0.3
                )
                
                if not has_hips and has_shoulders:
                    ls = draw_landmarks["LEFT_SHOULDER"]
                    rs = draw_landmarks["RIGHT_SHOULDER"]
                    dx = ls[0] - rs[0]
                    dy = ls[1] - rs[1]
                    S = math.sqrt(dx * dx + dy * dy)
                    if S > 1e-9:
                        rx, ry = dx / S, dy / S
                        ux, uy = ry, -rx
                        virtual_offset_x = 1.5 * S * ux
                        virtual_offset_y = 1.5 * S * uy
                        draw_landmarks["LEFT_HIP"] = (
                            ls[0] - virtual_offset_x,
                            ls[1] - virtual_offset_y,
                            -1.0,  # Negative visibility signals virtual landmark
                        )
                        draw_landmarks["RIGHT_HIP"] = (
                            rs[0] - virtual_offset_x,
                            rs[1] - virtual_offset_y,
                            -1.0,
                        )

                # Connections definitions: (start, end)
                connections = [
                    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
                    ("LEFT_SHOULDER", "LEFT_ELBOW"),
                    ("LEFT_ELBOW", "LEFT_WRIST"),
                    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
                    ("RIGHT_ELBOW", "RIGHT_WRIST"),
                    ("LEFT_SHOULDER", "LEFT_HIP"),
                    ("RIGHT_SHOULDER", "RIGHT_HIP"),
                    ("LEFT_HIP", "RIGHT_HIP"),
                    ("LEFT_HIP", "LEFT_KNEE"),
                    ("LEFT_KNEE", "LEFT_ANKLE"),
                    ("RIGHT_HIP", "RIGHT_KNEE"),
                    ("RIGHT_KNEE", "RIGHT_ANKLE"),
                    ("NOSE", "LEFT_EAR"),
                    ("NOSE", "RIGHT_EAR"),
                ]

                # Draw skeleton lines
                for start_name, end_name in connections:
                    if start_name in draw_landmarks and end_name in draw_landmarks:
                        x1, y1, v1 = draw_landmarks[start_name]
                        x2, y2, v2 = draw_landmarks[end_name]
                        
                        p1 = (int(x1 * W), int(y1 * H))
                        p2 = (int(x2 * W), int(y2 * H))

                        # Color: yellow if either is virtual, green if both visible, red otherwise
                        if v1 == -1.0 or v2 == -1.0:
                            color = (0, 255, 255)  # Yellow BGR
                        elif v1 >= 0.3 and v2 >= 0.3:
                            color = (106, 206, 158)  # Green BGR
                        else:
                            color = (142, 118, 247)  # Red BGR
                            
                        cv2.line(visual_image, p1, p2, color, 2)

                # Draw joint circles
                for name, lm_data in draw_landmarks.items():
                    x, y, v = lm_data
                    px, py = int(x * W), int(y * H)
                    
                    if v == -1.0:
                        color = (0, 255, 255)  # Yellow
                    elif v >= 0.3:
                        color = (106, 206, 158)  # Green
                    else:
                        color = (142, 118, 247)  # Red
                        
                    cv2.circle(visual_image, (px, py), 5, color, -1)
                    cv2.circle(visual_image, (px, py), 6, (0, 0, 0), 1)  # black border for contrast
            else:
                # Pose detection failed HUD message
                cv2.putText(
                    visual_image,
                    "NO POSE DETECTED",
                    (320, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (142, 118, 247),
                    2,
                    cv2.LINE_AA,
                )

            # Draw HUD panel background (semi-transparent)
            overlay = visual_image.copy()
            cv2.rectangle(overlay, (10, 10), (290, 200), (30, 27, 26), -1)
            cv2.addWeighted(overlay, 0.8, visual_image, 0.2, 0, visual_image)
            cv2.rectangle(visual_image, (10, 10), (290, 200), (247, 162, 122), 1)

            # Draw HUD Text
            # Title
            cv2.putText(
                visual_image,
                "AUTO ANNOTATOR",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (247, 162, 122),
                2,
                cv2.LINE_AA,
            )
            # State
            state_str = f"State: {self.state.name}"
            cv2.putText(
                visual_image,
                state_str,
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            # Window
            window_str = f"Window: {_format_time_window(self._window_index, 10)}"
            cv2.putText(
                visual_image,
                window_str,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            # Divider
            cv2.line(visual_image, (20, 85), (280, 85), (137, 95, 86), 1)

            # Feature values
            y_offset = 105
            features_info = [
                ("Body", reading.get("body")),
                ("Head", reading.get("head")),
                ("Hand", reading.get("hand")),
                ("Leg", reading.get("leg")),
            ]
            for name, val in features_info:
                label = f"{name}: "
                cv2.putText(
                    visual_image,
                    label,
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                if val is None:
                    val_str = "NA"
                    color = (137, 95, 86)  # grey
                elif val == -1:
                    val_str = "Left"
                    color = (142, 118, 247)  # red
                elif val == 1:
                    val_str = "Right"
                    color = (106, 206, 158)  # green
                else:
                    val_str = "Center"
                    color = (247, 162, 122)  # blue/accent

                cv2.putText(
                    visual_image,
                    val_str,
                    (80, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                y_offset += 22

            # Show the frame
            cv2.imshow("Auto Annotator — Pose Tracking", visual_image)
            cv2.waitKey(1)

        except Exception as e:
            logger.warning("Failed to render visualization overlay: %s", e, exc_info=True)

    def _save_debug_frame(self, image) -> None:
        """Save a frame to disk for diagnostic inspection."""
        try:
            import cv2
            debug_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'debug',
            )
            os.makedirs(debug_dir, exist_ok=True)
            path = os.path.join(debug_dir, f"frame_{self._debug_frame_count}.jpg")
            cv2.imwrite(path, image)
            logger.info(
                "DEBUG: Saved frame %d → %s  (shape=%s, mean=%.1f)",
                self._debug_frame_count, path, image.shape, float(image.mean()),
            )
            self._debug_frame_count += 1
        except Exception as e:
            logger.warning("Failed to save debug frame: %s", e)

    # ── Calibration phase ──────────────────────────────────────────────

    def _handle_calibration_tick(self, metrics: FrameMetrics) -> None:
        """Process one calibration second."""
        self._calibrator.add_reading(metrics)
        self._calibration_metrics.append(metrics)

        if self._calibrator.is_complete:
            self._thresholds = self._calibrator.compute_thresholds()
            self.state = SessionState.RUNNING
            logger.info("Calibration complete → entering main loop.")

            # Classify all collected calibration metrics retrospectively
            for cal_metrics in self._calibration_metrics:
                reading = classify_frame(cal_metrics, self._thresholds)
                self._handle_running_tick(reading)
            self._calibration_metrics.clear()

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

        # Write to checkpoint file and Link Queue sheet immediately
        self._save_checkpoint()
        if self._queue_mgr:
            last_sec = self._window_index * 10 + len(self._window_readings) - 1
            try:
                self._queue_mgr.mark_progress(self._video_id, last_sec)
            except Exception as e:
                logger.error("Failed to write progress to queue sheet: %s", e)

        # Reset for next window.
        self._window_readings = []
        self._window_index += 1

    def _save_checkpoint(self) -> None:
        """Write current window's raw per-second observations to the JSON checkpoint file."""
        from datetime import datetime
        checkpoint_data = None
        if self._checkpoint_path.exists():
            try:
                with open(self._checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
            except Exception as e:
                logger.error("Failed to load existing checkpoint: %s", e)

        if not checkpoint_data:
            duration = getattr(self._stream_reader, "duration", 0.0)
            total_windows = int(math.ceil(duration / 10.0)) if duration > 0 else 0
            checkpoint_data = {
                "session_info": {
                    "video_url": self._video_url,
                    "video_id": self._video_id,
                    "category": "Podcast",
                    "total_duration": duration,
                    "total_windows": total_windows,
                    "annotator": "",
                    "created_at": datetime.now().isoformat(),
                    "last_modified": datetime.now().isoformat()
                },
                "windows": []
            }

        # Update thresholds
        if self._thresholds:
            checkpoint_data["session_info"]["thresholds"] = {
                "T_body": self._thresholds.T_body,
                "T_head": self._thresholds.T_head,
                "T_hand": self._thresholds.T_hand,
                "T_leg": self._thresholds.T_leg,
            }

        start_sec = self._window_index * 10
        end_sec = start_sec + 10

        obs_list = []
        for offset, reading in enumerate(self._window_readings):
            obs_list.append({
                "time_sec": start_sec + offset,
                "hand": reading.get("hand"),
                "leg": reading.get("leg"),
                "head": reading.get("head"),
                "body": reading.get("body")
            })

        # Pad partial windows with null observations to maintain schema structure
        while len(obs_list) < 10:
            offset = len(obs_list)
            obs_list.append({
                "time_sec": start_sec + offset,
                "hand": None,
                "leg": None,
                "head": None,
                "body": None
            })

        new_window = {
            "window_id": self._window_index,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "observations": obs_list
        }

        windows = checkpoint_data.setdefault("windows", [])
        found_idx = -1
        for idx, win in enumerate(windows):
            if win.get("window_id") == self._window_index:
                found_idx = idx
                break

        if found_idx >= 0:
            windows[found_idx] = new_window
        else:
            windows.append(new_window)

        windows.sort(key=lambda w: w.get("window_id", 0))
        checkpoint_data["session_info"]["last_modified"] = datetime.now().isoformat()

        try:
            self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self._checkpoint_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2)

            if self._checkpoint_path.exists():
                backup_path = self._checkpoint_path.with_suffix(".backup.json")
                if backup_path.exists():
                    backup_path.unlink()
                self._checkpoint_path.rename(backup_path)

            temp_path.rename(self._checkpoint_path)
            logger.info("Saved raw checkpoint to %s", self._checkpoint_path.name)
        except Exception as e:
            logger.error("Failed to write checkpoint file: %s", e)

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
            "current_time": self._stream_reader.current_pts,
            "video_duration": time.time() - self._session_start_time,
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

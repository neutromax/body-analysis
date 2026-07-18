"""
Auto Annotator — PyAV In-Memory Stream Reader
==============================================

Uses yt-dlp to resolve a YouTube URL to a direct video streaming URL,
and PyAV (av) to decode and sample frames directly in-memory at 1 FPS.

This eliminates the need for screen-capture / BitBlt coordinates, and
guarantees compatibility in headless or sandboxed environments.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import av
import cv2
import numpy as np
import yt_dlp

logger = logging.getLogger(__name__)


class StreamReader:
    """
    Decodes a YouTube video stream in-memory using PyAV.
    Runs a background thread to decode and cache the current video frame.
    """

    def __init__(self, youtube_url: str, get_player_time_fn: Optional[callable] = None) -> None:
        """
        Args:
            youtube_url:        The YouTube video URL.
            get_player_time_fn: A function that returns the player's current playback time in seconds.
        """
        self.youtube_url = youtube_url
        self.get_player_time = get_player_time_fn

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Thread-safe properties for the parent process/caller
        self._frame_lock = threading.Lock()
        self._current_frame: Optional[np.ndarray] = None
        self._current_pts: float = -1.0

        self._is_resolved = threading.Event()
        self._resolve_error: Optional[str] = None

        self._is_finished = threading.Event()
        self._decode_start_time: float = 0.0

    @property
    def current_frame(self) -> Optional[np.ndarray]:
        """Get the latest decoded frame as a BGR NumPy array."""
        with self._frame_lock:
            return self._current_frame

    @property
    def current_pts(self) -> float:
        """Get the PTS (playback timestamp) in seconds of the current frame."""
        with self._frame_lock:
            return self._current_pts

    @property
    def is_finished(self) -> bool:
        """Whether the stream has finished decoding all frames."""
        return self._is_finished.is_set()

    def start(self) -> None:
        """Start the background decoding thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background decoding thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Stream reader stopped.")

    def _resolve_stream_url(self) -> Optional[tuple]:
        """
        Resolve YouTube URL to direct video stream URL using yt-dlp.

        Returns:
            (stream_url, http_headers_str) or None on failure.
        """
        ydl_opts = {
            "format": "bestvideo[height<=720][ext=mp4]/best[height<=720][ext=mp4]/bestvideo[ext=mp4]/best[ext=mp4]/best",
            "quiet": True,
            "no_warnings": True,
        }
        logger.info("Resolving YouTube stream URL...")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.youtube_url, download=False)
                url = info["url"]

                # Build HTTP headers string for FFmpeg/PyAV.
                http_headers = info.get("http_headers", {})
                headers_str = ""
                for k, v in http_headers.items():
                    headers_str += f"{k}: {v}\r\n"

                logger.info(
                    "Resolved stream URL (format=%s, resolution=%sx%s)",
                    info.get("format_id", "?"),
                    info.get("width", "?"),
                    info.get("height", "?"),
                )
                return url, headers_str
        except Exception as e:
            self._resolve_error = str(e)
            logger.error("Failed to resolve stream URL: %s", e)
            return None

    def _run(self) -> None:
        """Background thread: resolve URL, open PyAV container, decode frames."""
        result = self._resolve_stream_url()
        if not result:
            self._is_finished.set()
            return

        stream_url, headers_str = result
        self._is_resolved.set()

        container = None
        video_stream = None

        def _open_container(seek_time: float = 0.0) -> bool:
            nonlocal container, video_stream
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
            try:
                logger.info("Opening PyAV container (seek=%.2fs)...", seek_time)
                options = {
                    "timeout": "15000000",
                    "reconnect": "1",
                    "reconnect_streamed": "1",
                    "reconnect_delay_max": "5",
                }
                if headers_str:
                    options["headers"] = headers_str

                container = av.open(stream_url, options=options)
                video_stream = container.streams.video[0]
                video_stream.thread_type = "AUTO"
                if seek_time > 0.0:
                    pts = int(seek_time / video_stream.time_base)
                    container.seek(pts, stream=video_stream)
                return True
            except Exception as e:
                logger.error("Failed to open stream container: %s", e)
                return False

        if not _open_container(0.0):
            self._is_finished.set()
            return

        logger.info("In-memory PyAV decoding loop started.")
        last_cache_time = 0.0
        self._decode_start_time = time.time()

        while not self._stop_event.is_set():
            try:
                for frame in container.decode(video=0):
                    if self._stop_event.is_set():
                        break

                    pts_sec = float(frame.pts * video_stream.time_base)

                    # Cache at most 1 frame per second (skip intermediate frames)
                    if pts_sec - last_cache_time < 0.9:
                        continue

                    bgr_img = np.ascontiguousarray(frame.to_ndarray(format="bgr24"))
                    with self._frame_lock:
                        self._current_frame = bgr_img
                        self._current_pts = pts_sec
                    last_cache_time = pts_sec

                    # Pace to real-time so session controller ticks stay in sync.
                    elapsed_real = time.time() - self._decode_start_time
                    video_ahead = pts_sec - elapsed_real
                    if video_ahead > 0.5:
                        target = self._decode_start_time + pts_sec
                        while time.time() < target and not self._stop_event.is_set():
                            time.sleep(min(target - time.time(), 0.1))

                # If we exit the for loop normally, the stream ended
                logger.info("Stream decoding reached end of video.")
                break

            except Exception as e:
                logger.warning("PyAV decode error: %s. Retrying...", e)
                time.sleep(2.0)
                with self._frame_lock:
                    reopen_time = max(0.0, self._current_pts)
                if not _open_container(reopen_time):
                    break

        # Signal that decoding is complete.
        self._is_finished.set()

        # Cleanup
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        logger.info("In-memory PyAV decoding loop stopped.")


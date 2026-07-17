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

    def _resolve_stream_url(self) -> Optional[str]:
        """Resolve YouTube URL to direct video stream URL using yt-dlp."""
        ydl_opts = {
            "format": "bestvideo[ext=mp4]/best[ext=mp4]/best",
            "quiet": True,
            "no_warnings": True,
        }
        logger.info("Resolving YouTube stream URL...")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.youtube_url, download=False)
                return info["url"]
        except Exception as e:
            self._resolve_error = str(e)
            logger.error("Failed to resolve stream URL: %s", e)
            return None

    def _run(self) -> None:
        """Background thread logic for sequential decoding and time sync."""
        stream_url = self._resolve_stream_url()
        if not stream_url:
            return
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
                logger.info("Opening PyAV stream container (seek_time=%.2fs)...", seek_time)
                # Open with standard timeouts to prevent hangs
                container = av.open(stream_url, options={"timeout": "15000000"})
                video_stream = container.streams.video[0]
                video_stream.thread_type = "AUTO"  # Enable multi-threaded decoding
                if seek_time > 0.0:
                    pts = int(seek_time / video_stream.time_base)
                    container.seek(pts, stream=video_stream)
                return True
            except Exception as e:
                logger.error("Failed to open stream container: %s", e)
                return False

        if not _open_container(0.0):
            return

        logger.info("In-memory PyAV stream decoding loop started.")
        
        while not self._stop_event.is_set():
            try:
                # 1. Sync decoding time with player time if possible
                player_time = 0.0
                if self.get_player_time:
                    player_time = self.get_player_time()

                # Get current decoded frame's timestamp
                with self._frame_lock:
                    curr_pts = self._current_pts

                # Check if we are running too far ahead of the video player (e.g. > 3 seconds)
                if self.get_player_time and curr_pts > player_time + 3.0:
                    time.sleep(0.1)
                    continue

                # Check if the player has seeked backward or significantly forward (> 10 seconds ahead of us)
                if self.get_player_time:
                    # Seek backward detected
                    if player_time < curr_pts - 1.5:
                        logger.info("Seek backward detected (player: %.2fs, stream: %.2fs). Seeking stream...", player_time, curr_pts)
                        _open_container(player_time)
                        continue
                    # Large seek forward detected
                    elif player_time > curr_pts + 10.0:
                        logger.info("Seek forward detected (player: %.2fs, stream: %.2fs). Seeking stream...", player_time, curr_pts)
                        _open_container(player_time)
                        continue

                # 2. Decode the next frame from container
                frame_decoded = False
                for frame in container.decode(video=0):
                    if self._stop_event.is_set():
                        break

                    pts_sec = float(frame.pts * video_stream.time_base)
                    
                    # Convert to standard BGR image in-memory
                    bgr_img = frame.to_ndarray(format="bgr24")

                    # Cache the frame
                    with self._frame_lock:
                        self._current_frame = bgr_img
                        self._current_pts = pts_sec

                    frame_decoded = True
                    
                    # Break out to check player sync again
                    break

                if not frame_decoded:
                    # End of stream or buffer empty, sleep briefly
                    time.sleep(0.1)

            except Exception as e:
                logger.warning("Error during PyAV stream decoding: %s. Reopening...", e)
                time.sleep(1.0)
                # Attempt container recovery/reopen at the last known time
                with self._frame_lock:
                    reopen_time = max(0.0, self._current_pts)
                _open_container(reopen_time)

        # Cleanup
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        logger.info("In-memory PyAV stream decoding loop stopped.")

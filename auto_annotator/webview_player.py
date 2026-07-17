"""
Auto Annotator — WebView Player
=================================

Manages an embedded YouTube player via pywebview (WebView2 on Windows).

Key design decisions:
    - Resolves threading conflicts: pywebview demands to be run on the main
      thread. To prevent blocking Tkinter's event loop, the pywebview player is
      run in a separate child Process (via multiprocessing).
    - Inter-Process Communication (IPC): State information (ready status,
      player state, error codes, current time, duration, window coordinates)
      is shared using a multiprocessing Manager dict.
    - Autoplay policy bypass: The YouTube player is loaded muted with playsinline.
    - Diagnostic logs: Thorough console.log integration on the JS side bridged
      to Python log messages. Devtools enabled (debug=True).
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Tuple

import webview

from auto_config import PLAYER_WIDTH, PLAYER_HEIGHT, PLAYER_TIMEOUT

logger = logging.getLogger(__name__)

# ── HTML template for the YouTube IFrame Player API ────────────────────

_PLAYER_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #000; overflow: hidden; }
  #player { width: 100vw; height: 100vh; }
</style>
</head>
<body>
<div id="player"></div>
<script>
  var eventQueue = [];
  var pywebviewReady = false;

  function sendToPython(method, data) {
    if (pywebviewReady && window.pywebview && window.pywebview.api && window.pywebview.api[method]) {
      window.pywebview.api[method](data);
    } else {
      console.log("Queueing event for Python: " + method + "(" + data + ")");
      eventQueue.push({ method: method, data: data });
    }
  }

  window.addEventListener('pywebviewready', function() {
    console.log("JS: pywebviewready event fired. Processing queue of length " + eventQueue.length);
    pywebviewReady = true;
    while (eventQueue.length > 0) {
      var evt = eventQueue.shift();
      if (window.pywebview.api[evt.method]) {
        window.pywebview.api[evt.method](evt.data);
      }
    }
  });

  function jsLog(msg) {
    console.log(msg);
    sendToPython('log_from_js', msg);
  }

  jsLog("JS: Script starting, injecting YouTube IFrame API");

  var player;
  var tag = document.createElement('script');
  tag.src = "https://www.youtube.com/iframe_api";
  var firstScriptTag = document.getElementsByTagName('script')[0];
  firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

  function onYouTubeIframeAPIReady() {
    jsLog("JS: onYouTubeIframeAPIReady fired");
    try {
      player = new YT.Player('player', {
        videoId: '__VIDEO_ID__',
        playerVars: {
          autoplay: 0,
          controls: 1,
          modestbranding: 1,
          rel: 0,
          playsinline: 1,
          mute: 1,
        },
        events: {
          onReady: onPlayerReady,
          onStateChange: onPlayerStateChange,
          onError: onPlayerError,
        }
      });
      jsLog("JS: YT.Player constructor completed successfully");
    } catch (e) {
      jsLog("JS: ERROR constructing YT.Player: " + e.message);
    }
  }

  function onPlayerReady(event) {
    jsLog("JS: onPlayerReady triggered");
    sendToPython('on_player_ready', null);
  }

  function onPlayerStateChange(event) {
    jsLog("JS: onPlayerStateChange triggered, new state = " + event.data);
    sendToPython('on_state_change', event.data);
  }

  function onPlayerError(event) {
    jsLog("JS: onPlayerError triggered, error code = " + event.data);
    sendToPython('on_player_error', event.data);
  }

  function getPlayerState() {
    if (player && player.getPlayerState) {
      return player.getPlayerState();
    }
    return -1;
  }

  function getCurrentTime() {
    if (player && player.getCurrentTime) {
      return player.getCurrentTime();
    }
    return 0;
  }

  function getDuration() {
    if (player && player.getDuration) {
      return player.getDuration();
    }
    return 0;
  }
</script>
</body>
</html>
"""


def _extract_video_id(url: str) -> str:
    """Extract 11-char video ID from various YouTube URL formats."""
    import re
    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|\/embed\/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


# ── JS API Bridge for pywebview Process ────────────────────────────────

class ChildProcessJsBridge:
    """
    Receives callbacks from JS inside the pywebview process and writes
    them to the shared state dictionary.
    """

    def __init__(self, shared_state: dict) -> None:
        self.shared_state = shared_state

    def log_from_js(self, message: str) -> None:
        """Relay JS console logs to Python's logger."""
        logger.info("[JS Console] %s", message)

    def on_player_ready(self) -> None:
        logger.info("Python (Child): Received player ready event from JS")
        self.shared_state["is_ready"] = True

    def on_state_change(self, state: int) -> None:
        logger.info("Python (Child): Received state change event from JS: %d", state)
        self.shared_state["state"] = state

    def on_player_error(self, error_code: int) -> None:
        logger.error("Python (Child): Received error event from JS: %d", error_code)
        self.shared_state["error_code"] = error_code


# ── Subprocess event loop target ───────────────────────────────────────

def _run_player_process(
    html: str,
    shared_state: dict,
    stop_event: multiprocessing.Event,
) -> None:
    """Runs as the main thread of the player subprocess."""
    # Configure logging inside the child process.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (ChildProcess) %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    bridge = ChildProcessJsBridge(shared_state)
    window = webview.create_window(
        title="Auto Annotator — Video",
        html=html,
        width=PLAYER_WIDTH,
        height=PLAYER_HEIGHT,
        js_api=bridge,
        resizable=True,
    )

    def _state_poll_loop() -> None:
        """Poll window bbox and JS time/duration from child background thread."""
        logger.info("Child status polling thread started.")
        while not stop_event.is_set():
            try:
                # Update window coordinates.
                shared_state["bbox"] = (window.x, window.y, window.width, window.height)

                # Query current time & duration.
                curr_time = window.evaluate_js("getCurrentTime()")
                if curr_time is not None:
                    shared_state["current_time"] = float(curr_time)

                duration = window.evaluate_js("getDuration()")
                if duration is not None:
                    shared_state["duration"] = float(duration)
            except Exception as e:
                # Window might have been closed or not fully initialized yet.
                pass
            time.sleep(0.2)
        logger.info("Child status polling thread stopping.")

    poll_thread = threading.Thread(target=_state_poll_loop, daemon=True)
    poll_thread.start()

    # Start the pywebview event loop on the main thread of the child process.
    # debug=True enables the Web Inspector / DevTools right-click menu.
    webview.start(debug=True)
    logger.info("Player process GUI loop terminated.")


# ── Parent GUI wrapper class ───────────────────────────────────────────

class ParentBridge:
    """
    Parent-side compatibility wrapper that mimics the old JsBridge interface.
    """
    def __init__(self) -> None:
        self._on_ended: Optional[Callable] = None

    def set_on_ended(self, callback: Callable) -> None:
        """Register a callback to be triggered when the video ends."""
        self._on_ended = callback

    def trigger_ended(self) -> None:
        """Invokes the registered ended callback."""
        if self._on_ended:
            try:
                self._on_ended()
            except Exception as e:
                logger.error("Error invoking parent-side ended callback: %s", e)


class WebViewPlayer:
    """
    Manages the player subprocess from the main Tkinter process.
    """

    def __init__(self) -> None:
        self._manager = multiprocessing.Manager()
        self._shared_state = self._manager.dict({
            "is_ready": False,
            "state": -1,
            "error_code": None,
            "current_time": 0.0,
            "duration": 0.0,
            "bbox": None,
        })
        self._stop_event = multiprocessing.Event()
        self._parent_stop_event = threading.Event()
        self._process: Optional[multiprocessing.Process] = None
        self._start_time: float = 0.0
        self._bridge = ParentBridge()
        self._parent_poll_thread: Optional[threading.Thread] = None

    @property
    def bridge(self) -> ParentBridge:
        """Exposes ParentBridge for API compatibility with main_app."""
        if self._bridge is None:
            raise RuntimeError("WebViewPlayer.bridge accessed but it is uninitialized or closed.")
        return self._bridge

    def load(self, youtube_url: str) -> None:
        """Spawn the child process running the player."""
        video_id = _extract_video_id(youtube_url)
        html = _PLAYER_HTML.replace("__VIDEO_ID__", video_id)
        self._start_time = time.time()
        self._stop_event.clear()
        self._parent_stop_event.clear()

        # Reset states.
        self._shared_state["is_ready"] = False
        self._shared_state["state"] = -1
        self._shared_state["error_code"] = None
        self._shared_state["current_time"] = 0.0
        self._shared_state["duration"] = 0.0
        self._shared_state["bbox"] = None

        logger.info("[loading] Spawning player subprocess for video ID: %s", video_id)

        self._process = multiprocessing.Process(
            target=_run_player_process,
            args=(html, self._shared_state, self._stop_event),
            daemon=True,
        )
        self._process.start()

        # Start a parent-side polling thread to monitor state changes and trigger ended callback
        def _parent_state_poll() -> None:
            logger.info("Parent-side state polling thread started.")
            last_state = -1
            while not self._parent_stop_event.is_set():
                try:
                    current_state = self._shared_state["state"]
                    if current_state == 0 and last_state != 0:
                        logger.info("Parent process detected video state: ENDED. Invoking on_ended callback.")
                        self._bridge.trigger_ended()
                    last_state = current_state
                except Exception:
                    pass
                time.sleep(0.5)
            logger.info("Parent-side state polling thread stopped.")

        self._parent_poll_thread = threading.Thread(target=_parent_state_poll, daemon=True)
        self._parent_poll_thread.start()

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        """Check if ready (mostly for compatibility with tests)."""
        start = time.time()
        while time.time() - start < timeout:
            if self._shared_state["is_ready"]:
                return True
            time.sleep(0.1)
        return False

    def get_status(self) -> str:
        """Query loading status from the shared dict."""
        if self._shared_state["is_ready"]:
            return "ready"

        err = self._shared_state["error_code"]
        if err is not None:
            logger.error("[error] Player process reported error code: %s", err)
            if err in (101, 150, 153):
                return "video not embeddable / restricted"
            elif err == 2:
                return "invalid parameters"
            elif err == 5:
                return "HTML5 player error"
            elif err == 100:
                return "video not found"
            else:
                return f"unknown player error (code {err})"

        # Check for timeout.
        elapsed = time.time() - self._start_time
        if elapsed > PLAYER_TIMEOUT:
            logger.error("[timeout] Load timed out after %ds", PLAYER_TIMEOUT)
            return f"timed out after {PLAYER_TIMEOUT}s"

        return "loading"

    def get_player_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Get window position from shared dict."""
        return self._shared_state["bbox"]

    def get_current_time(self) -> float:
        return self._shared_state["current_time"]

    def get_duration(self) -> float:
        return self._shared_state["duration"]

    def get_state(self) -> int:
        return self._shared_state["state"]

    def is_playing(self) -> bool:
        return self._shared_state["state"] == 1

    def is_ended(self) -> bool:
        return self._shared_state["state"] == 0

    def close(self) -> None:
        """Terminate the subprocess and release resources."""
        self._parent_stop_event.set()
        self._stop_event.set()
        if self._process is not None:
            if self._process.is_alive():
                logger.info("Terminating player subprocess...")
                self._process.terminate()
                self._process.join(timeout=2)
                if self._process.is_alive():
                    self._process.kill()
            self._process = None
        self._manager.shutdown()
        logger.info("WebView player process cleaned up.")

"""
Auto Annotator — Frame Capture
================================

Captures a screenshot of the embedded YouTube player's screen region
using the `mss` library.

Why screen capture?
    An embedded browser (pywebview/WebView2) playing YouTube does NOT
    expose raw decoded video frames to Python.  There is no official API
    for pulling frame buffers from a YouTube iframe.  The practical,
    honest way to get "the current frame" is to screenshot the player's
    screen region.

Performance:
    mss captures at ~30ms on modern hardware, well within the 1-second
    interval between captures.  The bottleneck is MediaPipe, not capture.

Usage:
    from frame_capture import capture_region
    image = capture_region((left, top, width, height))
    # image is a NumPy BGR array ready for pose_engine.detect()
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np
import mss

logger = logging.getLogger(__name__)

# Reusable mss instance (thread-safe for single-threaded capture).
_sct = None


def _get_sct() -> mss.mss:
    """Lazy-init a module-level mss instance."""
    global _sct
    if _sct is None:
        _sct = mss.mss()
    return _sct


def capture_region(
    bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Capture a screen region and return it as a NumPy BGR image.

    Args:
        bbox:  (left, top, width, height) in absolute screen pixels.
               This should be the bounding rectangle of the pywebview
               player window's content area.

    Returns:
        NumPy array of shape (height, width, 3), dtype uint8, in BGR
        color order (OpenCV convention).

    Raises:
        RuntimeError:  If the capture fails or returns an empty image.
    """
    left, top, width, height = bbox

    monitor = {
        "left": left,
        "top": top,
        "width": width,
        "height": height,
    }

    sct = _get_sct()
    screenshot = sct.grab(monitor)

    # mss returns BGRA; convert to BGR for OpenCV/MediaPipe.
    img = np.array(screenshot, dtype=np.uint8)

    if img.size == 0:
        raise RuntimeError(
            f"Screen capture returned an empty image for region {bbox}"
        )

    # Drop the alpha channel (BGRA → BGR).
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    logger.debug("Captured region %s → shape %s", bbox, img.shape)
    return img

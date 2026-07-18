"""
Auto Annotator — Pose Engine
==============================

Wraps MediaPipe Pose Landmarker to extract landmarks from a single image.

Single responsibility: take an image (NumPy BGR array), run pose
detection, return a structured dict of landmark names → (x, y, visibility).
All MediaPipe setup and teardown is encapsulated here.

Uses the MediaPipe Tasks API (PoseLandmarker), which is the current
supported API as of mediapipe ≥ 0.10.15.  The legacy mp.solutions.pose
was removed in newer versions.

The model file (pose_landmarker_full.task) is auto-downloaded on first
use and cached in the auto_annotator directory.

Usage:
    engine = PoseEngine()
    result = engine.detect(image)
    if result is not None:
        nose_x, nose_y, nose_vis = result["NOSE"]
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

from auto_config import LANDMARK, VISIBILITY_THRESHOLD

logger = logging.getLogger(__name__)

# Type alias: each landmark is (x, y, visibility).
LandmarkData = Dict[str, Tuple[float, float, float]]

# ── Model file management ──────────────────────────────────────────────

# Model files are stored next to this source file.
_MODEL_DIR = Path(__file__).parent
_MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
_MODEL_FILENAMES = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}


def _ensure_model(complexity: int) -> str:
    """
    Ensure the model file for the given complexity level exists locally.

    Downloads from Google's model hub on first use.  The file is cached
    in the auto_annotator directory for subsequent runs.

    Args:
        complexity:  0 = lite, 1 = full, 2 = heavy.

    Returns:
        Absolute path to the .task model file.

    Raises:
        ValueError:  If complexity is not 0, 1, or 2.
        RuntimeError: If the download fails.
    """
    if complexity not in _MODEL_URLS:
        raise ValueError(f"model_complexity must be 0, 1, or 2, got {complexity}")

    filename = _MODEL_FILENAMES[complexity]
    filepath = _MODEL_DIR / filename

    if filepath.exists():
        logger.info("Model file found: %s", filepath)
        return str(filepath)

    url = _MODEL_URLS[complexity]
    logger.info("Downloading model (complexity=%d) from %s ...", complexity, url)

    try:
        urllib.request.urlretrieve(url, str(filepath))
        logger.info("Model saved to: %s", filepath)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download pose landmarker model from {url}: {e}"
        ) from e

    return str(filepath)


class PoseEngine:
    """
    Stateful wrapper around MediaPipe PoseLandmarker (Tasks API).

    Creates the model once on construction and reuses it across frames.
    Call `close()` when done (or use as a context manager) to release
    resources.

    Attributes:
        _landmarker:  The underlying PoseLandmarker instance.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
    ) -> None:
        """
        Initialize the MediaPipe PoseLandmarker.

        Args:
            model_complexity:  0 = lite, 1 = full, 2 = heavy.
                               1 is a good speed/accuracy trade-off for
                               1 FPS capture from a screen recording.
            min_detection_confidence:  Minimum confidence for person
                                       detection to succeed.
            min_tracking_confidence:   Minimum confidence for landmark
                                       tracking between frames.
        """
        model_path = _ensure_model(model_complexity)

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms: int = -1

        logger.info(
            "PoseEngine initialized (complexity=%d, det=%.2f, track=%.2f, mode=VIDEO)",
            model_complexity,
            min_detection_confidence,
            min_tracking_confidence,
        )

    # ── Context manager support ────────────────────────────────────────

    def __enter__(self) -> "PoseEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        logger.info("PoseEngine closed.")

    # ── Core detection ─────────────────────────────────────────────────

    def detect(self, image: np.ndarray, timestamp_ms: int = 0) -> Optional[LandmarkData]:
        """
        Run pose detection on a single image using VIDEO mode tracking.

        Args:
            image:         A NumPy array in BGR format (as returned by
                           OpenCV or mss screen capture).  Shape: (H, W, 3).
            timestamp_ms:  Frame timestamp in milliseconds.  Must increase
                           monotonically between calls (enforced internally).

        Returns:
            A dict mapping landmark name → (x, y, visibility) in
            MediaPipe's normalized image space (0..1), or None if no
            person was detected.

            Only the landmarks listed in auto_config.LANDMARK are included.

        Raises:
            ValueError:  If the image has an unexpected shape.
        """
        if image is None or image.ndim != 3:
            raise ValueError(
                f"Expected a 3-channel image, got shape={getattr(image, 'shape', None)}"
            )

        # Ensure the array is contiguous (required by cv2 and MediaPipe).
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)

        # Ensure monotonically increasing timestamp for VIDEO mode.
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        # MediaPipe expects RGB; OpenCV/mss gives BGR.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Wrap in a MediaPipe Image.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Run detection with temporal tracking.
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            logger.debug(
                "No pose detected in frame (shape=%s, mean_px=%.0f).",
                image.shape, image.mean(),
            )
            return None

        # Use the first detected pose.
        pose_landmarks = result.pose_landmarks[0]

        # Extract only the landmarks we care about.
        landmarks: LandmarkData = {}
        for name, idx in LANDMARK.items():
            lm = pose_landmarks[idx]
            landmarks[name] = (lm.x, lm.y, lm.visibility)

        return landmarks

    # ── Utility ────────────────────────────────────────────────────────

    @staticmethod
    def are_landmarks_visible(
        landmarks: LandmarkData,
        required: List[str],
        threshold: float = VISIBILITY_THRESHOLD,
    ) -> bool:
        """
        Check whether ALL of the named landmarks meet the visibility
        threshold.

        This is the data-quality gate (Section 6.7): if any required
        landmark is below V_min, the feature's reading should be NA.

        Args:
            landmarks:  The full landmark dict from detect().
            required:   List of landmark names that must be visible.
            threshold:  Minimum visibility score.

        Returns:
            True if every required landmark's visibility >= threshold.
        """
        for name in required:
            if name not in landmarks:
                return False
            _, _, vis = landmarks[name]
            if vis < threshold:
                return False
        return True

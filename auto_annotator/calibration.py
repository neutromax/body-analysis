"""
Auto Annotator — Calibration Module
=====================================

Derives per-video classification thresholds from a neutral baseline,
rather than using fixed magic numbers.

How it works (Section 6.6):
    1. During the first K seconds of a video, the person is assumed to
       be in their normal resting posture.
    2. Each second, we compute the same raw metrics as during annotation:
       body lean angle, head lateral offset, wrist velocity, ankle velocity.
    3. For each metric, we compute its mean (μ) and standard deviation (σ)
       across the calibration window.
    4. The threshold is set to:  T = |μ| + k × σ
       This means "intentional movement" must exceed the person's own
       natural resting variation by k standard deviations.

Why this is better than fixed thresholds:
    - Adapts to the individual's resting posture (some people naturally
      sit slightly tilted).
    - Adapts to the camera setup (zoom level, angle).
    - The sensitivity parameter k is intuitive: "how many sigmas beyond
      resting counts as intentional."

Fallback:
    If calibration can't run (person not visible in the first K seconds,
    or fewer than 2 valid readings), we fall back to conservative fixed
    thresholds from config.py — clearly labeled as a fallback.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from auto_config import (
    CALIBRATION_DURATION,
    SIGMA_MULTIPLIER,
    FALLBACK_T_BODY,
    FALLBACK_T_HEAD,
    FALLBACK_T_HAND,
    FALLBACK_T_LEG,
)
from feature_classifier import FrameMetrics, Thresholds

logger = logging.getLogger(__name__)


def _mean_and_std(values: List[float]) -> tuple[float, float]:
    """
    Compute mean and population standard deviation of a list of floats.

    Returns (0.0, 0.0) if the list is empty.

    We use population std (ddof=0) because we're characterizing the
    ACTUAL variation observed during calibration, not estimating a
    population parameter from a sample.
    """
    if not values:
        return 0.0, 0.0
    n = len(values)
    mu = sum(values) / n
    variance = sum((v - mu) ** 2 for v in values) / n
    sigma = math.sqrt(variance)
    return mu, sigma


def _compute_threshold(
    values: List[float],
    k: float,
    fallback: float,
    metric_name: str,
) -> float:
    """
    Compute a single threshold from calibration readings.

    Formula:  T = |μ| + k × σ

    If fewer than 2 valid readings exist, returns the fallback value.

    Args:
        values:       List of metric readings during calibration.
        k:            Sigma multiplier (how many σ beyond resting = intentional).
        fallback:     Fallback threshold if calibration data is insufficient.
        metric_name:  For logging purposes.

    Returns:
        The computed threshold (always positive).
    """
    if len(values) < 2:
        logger.warning(
            "Calibration for '%s': only %d reading(s), using fallback=%.4f",
            metric_name, len(values), fallback,
        )
        return fallback

    mu, sigma = _mean_and_std(values)
    threshold = abs(mu) + k * sigma

    # Guard against a threshold of exactly zero (would make everything
    # intentional), which can happen if σ=0 and μ=0 (person perfectly still).
    # In that case, use fallback instead.
    if threshold < 1e-9:
        logger.warning(
            "Calibration for '%s': computed T=0 (μ=%.6f, σ=%.6f), using fallback=%.4f",
            metric_name, mu, sigma, fallback,
        )
        return fallback

    logger.info(
        "Calibration '%s': μ=%.4f, σ=%.4f → T=%.4f  (k=%.2f, n=%d)",
        metric_name, mu, sigma, threshold, k, len(values),
    )
    return threshold


class Calibrator:
    """
    Collects metric readings during the baseline period and produces
    per-video thresholds.

    Usage:
        cal = Calibrator()
        for each second during baseline:
            cal.add_reading(metrics)
        thresholds = cal.compute_thresholds()
    """

    def __init__(
        self,
        duration: int = CALIBRATION_DURATION,
        k: float = SIGMA_MULTIPLIER,
    ) -> None:
        """
        Args:
            duration:  Number of seconds for calibration (K).
            k:         Sigma multiplier for threshold computation.
        """
        self.duration = duration
        self.k = k

        # Collected readings per metric.
        self._body_angles: List[float] = []
        self._head_offsets: List[float] = []
        self._hand_velocities: List[float] = []
        self._leg_velocities: List[float] = []

        self._readings_count = 0

    @property
    def is_complete(self) -> bool:
        """Whether we've collected enough seconds of data."""
        return self._readings_count >= self.duration

    @property
    def readings_collected(self) -> int:
        """How many seconds of data we've collected so far."""
        return self._readings_count

    def add_reading(self, metrics: FrameMetrics) -> None:
        """
        Record one second's worth of raw metrics.

        Only non-None values are added to each metric's collection.
        For velocity metrics (hand, leg), we pick the dominant side
        (whichever moved more) — same as during classification.

        Args:
            metrics:  Raw FrameMetrics from compute_frame_metrics().
        """
        self._readings_count += 1

        if metrics.body_angle_deg is not None:
            self._body_angles.append(metrics.body_angle_deg)

        if metrics.head_local_x is not None:
            self._head_offsets.append(metrics.head_local_x)

        # For velocity: pick the dominant (larger |v|) side.
        hand_v = self._pick_dominant(metrics.hand_v_left, metrics.hand_v_right)
        if hand_v is not None:
            self._hand_velocities.append(hand_v)

        leg_v = self._pick_dominant(metrics.leg_v_left, metrics.leg_v_right)
        if leg_v is not None:
            self._leg_velocities.append(leg_v)

    @staticmethod
    def _pick_dominant(
        v_left: Optional[float],
        v_right: Optional[float],
    ) -> Optional[float]:
        """Pick whichever side moved more (or the only available one)."""
        if v_left is None and v_right is None:
            return None
        if v_left is None:
            return v_right
        if v_right is None:
            return v_left
        return v_left if abs(v_left) >= abs(v_right) else v_right

    def compute_thresholds(self) -> Thresholds:
        """
        Compute thresholds from collected calibration data.

        For each metric:  T = |μ| + k × σ

        Falls back to config.FALLBACK_T_* if insufficient data.

        Returns:
            A Thresholds dataclass ready for use by feature_classifier.
        """
        T_body = _compute_threshold(
            self._body_angles, self.k, FALLBACK_T_BODY, "body_lean"
        )
        T_head = _compute_threshold(
            self._head_offsets, self.k, FALLBACK_T_HEAD, "head_direction"
        )
        T_hand = _compute_threshold(
            self._hand_velocities, self.k, FALLBACK_T_HAND, "hand_movement"
        )
        T_leg = _compute_threshold(
            self._leg_velocities, self.k, FALLBACK_T_LEG, "leg_movement"
        )

        calibrated = (
            len(self._body_angles) >= 2
            and len(self._head_offsets) >= 2
        )

        logger.info(
            "Calibration complete: calibrated=%s, T_body=%.3f, T_head=%.4f, "
            "T_hand=%.4f, T_leg=%.4f",
            calibrated, T_body, T_head, T_hand, T_leg,
        )

        return Thresholds(
            T_body=T_body,
            T_head=T_head,
            T_hand=T_hand,
            T_leg=T_leg,
            calibrated=calibrated,
        )

    def get_fallback_thresholds(self) -> Thresholds:
        """
        Return fallback thresholds directly (no calibration).

        Used when the caller knows calibration can't or shouldn't run.
        """
        return Thresholds(
            T_body=FALLBACK_T_BODY,
            T_head=FALLBACK_T_HEAD,
            T_hand=FALLBACK_T_HAND,
            T_leg=FALLBACK_T_LEG,
            calibrated=False,
        )

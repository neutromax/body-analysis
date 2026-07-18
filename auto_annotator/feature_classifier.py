"""
Auto Annotator — Feature Classifier
=====================================

The ONLY module that turns geometry numbers into -1 / 0 / 1 labels.

Every classification decision lives here — nowhere else.  The geometry
module computes raw numbers; this module applies thresholds to produce
categorical labels.  This separation means thresholds are inspectable
and swappable without touching the math.

Each classify_* function implements the three-way decision from Section 6:
    value >  +T  →  1  (Right)
    value <  -T  → -1  (Left)
    else         →  0  (Center / Neutral)

The main entry point is `classify_frame()`, which orchestrates all four
features for one second of data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from auto_config import (
    VISIBILITY_THRESHOLD,
    HEAD_LANDMARK_MODE,
)
from geometry import (
    body_reference_frame,
    torso_lean_angle,
    head_lateral_offset,
    torso_local_position,
    midpoint,
    limb_lateral_velocity,
    euclidean_dist,
    unit_vector,
)
# Type alias: each landmark is (x, y, visibility).
# Defined here (not imported from pose_engine) to avoid dragging mediapipe
# into modules that only need the type for function signatures.
LandmarkData = Dict[str, Tuple[float, float, float]]

logger = logging.getLogger(__name__)


@dataclass
class Thresholds:
    """
    The four thresholds used to classify each feature.

    Produced by calibration.py (per-video, statistical) or set to
    fallback values from config.py.
    """
    T_body: float
    T_head: float
    T_hand: float
    T_leg: float
    calibrated: bool = False   # True if derived from calibration data


# Type alias for a single second's reading.
# Each feature is -1, 0, 1, or None (= NA, visibility too low).
FrameReading = Dict[str, Optional[int]]


# ───────────────────────────────────────────────────────────────────────
# Individual feature classifiers
# ───────────────────────────────────────────────────────────────────────

def classify_body_lean(angle_deg: float, threshold: float) -> int:
    """
    Classify body lean from the torso-to-vertical angle.

    Args:
        angle_deg:  Signed angle in degrees (positive = right lean).
        threshold:  Degrees beyond which lean is considered intentional.

    Returns:
        1 if leaning right, -1 if leaning left, 0 if within threshold.
    """
    if angle_deg > threshold:
        return 1
    if angle_deg < -threshold:
        return -1
    return 0


def classify_head_direction(local_x_head: float, threshold: float) -> int:
    """
    Classify head direction from the head's body-relative lateral offset.

    Args:
        local_x_head:  Lateral offset of head from shoulder center,
                       measured along the body's right axis, normalized
                       by shoulder width.
        threshold:     Offset beyond which the head is considered turned.

    Returns:
        1 if head is to body's right, -1 if left, 0 if center.
    """
    if local_x_head > threshold:
        return 1
    if local_x_head < -threshold:
        return -1
    return 0


def classify_limb_movement(
    v_left: Optional[float],
    v_right: Optional[float],
    threshold: float,
) -> Optional[int]:
    """
    Classify hand or leg movement from lateral velocities.

    Picks the DOMINANT side (whichever wrist/ankle moved more this
    second) and classifies its velocity.

    Args:
        v_left:    Lateral velocity of the left wrist/ankle (or None if
                   that landmark wasn't visible).
        v_right:   Lateral velocity of the right wrist/ankle (or None).
        threshold: Velocity beyond which movement is considered intentional.

    Returns:
        1 if dominant side moved right, -1 if left, 0 if neutral,
        or None if both sides are unavailable.
    """
    # Handle cases where one or both sides are unavailable.
    if v_left is None and v_right is None:
        return None
    if v_left is None:
        v_dominant = v_right
    elif v_right is None:
        v_dominant = v_left
    else:
        # Pick whichever moved more (dominant side).
        v_dominant = v_left if abs(v_left) >= abs(v_right) else v_right

    if v_dominant > threshold:
        return 1
    if v_dominant < -threshold:
        return -1
    return 0


def classify_limb_position(
    local_x_left: Optional[float],
    local_x_right: Optional[float],
    threshold: float = 0.15,
) -> Optional[int]:
    """
    Classify hand or leg **position** as a fallback when velocity is
    unavailable (first frame, or after a detection gap).

    Uses the lateral position of the dominant (most-displaced) side's
    wrist or ankle relative to the body center.

    Args:
        local_x_left:   Body-relative lateral position of left wrist/ankle.
        local_x_right:  Body-relative lateral position of right wrist/ankle.
        threshold:      Displacement beyond which position is significant
                        (in shoulder widths, default 0.15).
    Returns:
        1 if dominant side is to body's right, -1 if left, 0 if centered,
        None if both positions are unavailable.
    """
    if local_x_left is None and local_x_right is None:
        return None
    if local_x_left is None:
        dominant = local_x_right
    elif local_x_right is None:
        dominant = local_x_left
    else:
        dominant = local_x_left if abs(local_x_left) >= abs(local_x_right) else local_x_right

    if dominant > threshold:
        return 1
    if dominant < -threshold:
        return -1
    return 0


# ───────────────────────────────────────────────────────────────────────
# Frame-level orchestration
# ───────────────────────────────────────────────────────────────────────

# Landmarks required for each feature's formula (used for visibility gating).
REQUIRED_LANDMARKS: Dict[str, list] = {
    "body": ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"],
    "head": ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"],
    "hand": ["LEFT_WRIST", "RIGHT_WRIST", "LEFT_SHOULDER", "RIGHT_SHOULDER",
             "LEFT_HIP", "RIGHT_HIP"],
    "leg":  ["LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
             "LEFT_HIP", "RIGHT_HIP"],
}

# If using ear midpoint for head, replace NOSE with ears.
if HEAD_LANDMARK_MODE == "EAR_MIDPOINT":
    REQUIRED_LANDMARKS["head"] = [
        "LEFT_EAR", "RIGHT_EAR",
        "LEFT_SHOULDER", "RIGHT_SHOULDER",
        "LEFT_HIP", "RIGHT_HIP",
    ]


def _check_visibility(
    landmarks: LandmarkData,
    required: list[str],
    threshold: float,
) -> bool:
    """Check all required landmarks meet visibility threshold."""
    for name in required:
        if name not in landmarks:
            return False
        _, _, vis = landmarks[name]
        if vis < threshold:
            return False
    return True


def _get_xy(landmarks: LandmarkData, name: str) -> Tuple[float, float]:
    """Extract (x, y) from a landmark, discarding visibility."""
    x, y, _ = landmarks[name]
    return (x, y)


@dataclass
class FrameMetrics:
    """
    Raw numeric metrics computed for a single frame, BEFORE classification.

    These are what calibration.py collects during the baseline period, and
    what feature_classifier uses to produce -1/0/1.
    """
    body_angle_deg: Optional[float] = None     # torso lean angle
    head_local_x: Optional[float] = None       # head lateral offset
    hand_v_left: Optional[float] = None        # left wrist velocity
    hand_v_right: Optional[float] = None       # right wrist velocity
    leg_v_left: Optional[float] = None         # left ankle velocity
    leg_v_right: Optional[float] = None        # right ankle velocity

    # Body frame data needed for velocity computation in the next frame.
    wrist_left_local_x: Optional[float] = None
    wrist_right_local_x: Optional[float] = None
    ankle_left_local_x: Optional[float] = None
    ankle_right_local_x: Optional[float] = None


def compute_frame_metrics(
    landmarks: LandmarkData,
    prev_metrics: Optional[FrameMetrics],
    vis_threshold: float = VISIBILITY_THRESHOLD,
) -> FrameMetrics:
    """
    Compute all raw numeric metrics for one frame.

    This function bridges pose_engine's landmark output and the
    classification step.  It calls geometry.py functions to produce
    the numbers that classify_* functions will threshold.

    Args:
        landmarks:     Landmark dict from PoseEngine.detect().
        prev_metrics:  Metrics from the previous second (needed for
                       velocity computation).  None on the first frame.
        vis_threshold: Minimum visibility to trust a landmark.

    Returns:
        FrameMetrics with all computable values filled in, and None for
        any metric whose required landmarks weren't visible.
    """
    metrics = FrameMetrics()

    # ── Body frame (needed by almost everything) ───────────────────────
    body_visible = _check_visibility(
        landmarks, REQUIRED_LANDMARKS["body"], vis_threshold
    )

    if not body_visible:
        # Check if shoulders are visible for fallback
        shoulders_visible = _check_visibility(
            landmarks, ["LEFT_SHOULDER", "RIGHT_SHOULDER"], vis_threshold
        )
        if not shoulders_visible:
            return metrics  # Can't compute anything without at least shoulders.

        ls = _get_xy(landmarks, "LEFT_SHOULDER")
        rs = _get_xy(landmarks, "RIGHT_SHOULDER")
        S = euclidean_dist(ls, rs)
        if S < 1e-9:
            return metrics

        # r points from right shoulder to left shoulder (screen-right)
        v_sh = (ls[0] - rs[0], ls[1] - rs[1])
        r = unit_vector(v_sh)

        # u is perpendicular to r, pointing screen-up (negative y)
        u = (r[1], -r[0])

        # Estimate virtual hips: C_h = C_s - 1.5 * S * u
        # Since u points up, subtracting it goes DOWN (increasing y)
        virtual_offset = (1.5 * S * u[0], 1.5 * S * u[1])
        lh = (ls[0] - virtual_offset[0], ls[1] - virtual_offset[1])
        rh = (rs[0] - virtual_offset[0], rs[1] - virtual_offset[1])

        # Compute reference frame using the virtual hips
        C_s, C_h, S, u, r = body_reference_frame(ls, rs, lh, rh)

        # Exclude hips from required landmarks for fallback checks
        req_head = (
            ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER"]
            if HEAD_LANDMARK_MODE != "EAR_MIDPOINT"
            else ["LEFT_EAR", "RIGHT_EAR", "LEFT_SHOULDER", "RIGHT_SHOULDER"]
        )
        req_hand = ["LEFT_WRIST", "RIGHT_WRIST", "LEFT_SHOULDER", "RIGHT_SHOULDER"]
        req_leg = []  # Legs always require actual hips, so legs will remain None
    else:
        ls = _get_xy(landmarks, "LEFT_SHOULDER")
        rs = _get_xy(landmarks, "RIGHT_SHOULDER")
        lh = _get_xy(landmarks, "LEFT_HIP")
        rh = _get_xy(landmarks, "RIGHT_HIP")

        C_s, C_h, S, u, r = body_reference_frame(ls, rs, lh, rh)

        req_head = REQUIRED_LANDMARKS["head"]
        req_hand = REQUIRED_LANDMARKS["hand"]
        req_leg = REQUIRED_LANDMARKS["leg"]

    if S < 1e-9:
        return metrics  # Degenerate: shoulders overlap.

    # ── Body lean ──────────────────────────────────────────────────────
    metrics.body_angle_deg = torso_lean_angle(ls, rs, lh, rh)

    # ── Head direction ─────────────────────────────────────────────────
    head_visible = _check_visibility(
        landmarks, req_head, vis_threshold
    )
    if head_visible:
        if HEAD_LANDMARK_MODE == "EAR_MIDPOINT":
            le = _get_xy(landmarks, "LEFT_EAR")
            re = _get_xy(landmarks, "RIGHT_EAR")
            head_pt = midpoint(le, re)
        else:
            head_pt = _get_xy(landmarks, "NOSE")

        metrics.head_local_x = head_lateral_offset(head_pt, C_s, r, S)

    # ── Wrist positions (for hand velocity) ────────────────────────────
    hand_visible = _check_visibility(
        landmarks, req_hand, vis_threshold
    )
    if hand_visible:
        lw = _get_xy(landmarks, "LEFT_WRIST")
        rw = _get_xy(landmarks, "RIGHT_WRIST")

        lw_local_x, _ = torso_local_position(lw, C_s, u, r, S)
        rw_local_x, _ = torso_local_position(rw, C_s, u, r, S)

        metrics.wrist_left_local_x = lw_local_x
        metrics.wrist_right_local_x = rw_local_x

        # Velocity requires a previous frame.
        if prev_metrics is not None:
            if prev_metrics.wrist_left_local_x is not None:
                metrics.hand_v_left = limb_lateral_velocity(
                    lw_local_x, prev_metrics.wrist_left_local_x
                )
            if prev_metrics.wrist_right_local_x is not None:
                metrics.hand_v_right = limb_lateral_velocity(
                    rw_local_x, prev_metrics.wrist_right_local_x
                )

    # ── Ankle positions (for leg velocity) ─────────────────────────────
    if req_leg:
        leg_visible = _check_visibility(
            landmarks, req_leg, vis_threshold
        )
        if leg_visible:
            la = _get_xy(landmarks, "LEFT_ANKLE")
            ra = _get_xy(landmarks, "RIGHT_ANKLE")

            la_local_x, _ = torso_local_position(la, C_s, u, r, S)
            ra_local_x, _ = torso_local_position(ra, C_s, u, r, S)

            metrics.ankle_left_local_x = la_local_x
            metrics.ankle_right_local_x = ra_local_x

            if prev_metrics is not None:
                if prev_metrics.ankle_left_local_x is not None:
                    metrics.leg_v_left = limb_lateral_velocity(
                        la_local_x, prev_metrics.ankle_left_local_x
                    )
                if prev_metrics.ankle_right_local_x is not None:
                    metrics.leg_v_right = limb_lateral_velocity(
                        ra_local_x, prev_metrics.ankle_right_local_x
                    )

    return metrics


def classify_frame(
    metrics: FrameMetrics,
    thresholds: Thresholds,
) -> FrameReading:
    """
    Classify all four features for one second of data.

    This is the main entry point called by session_controller every second.

    Args:
        metrics:     Raw numeric metrics from compute_frame_metrics().
        thresholds:  Per-video thresholds from calibration.py.

    Returns:
        Dict with keys "hand", "leg", "head", "body", each mapped to
        -1, 0, 1, or None (NA if the metric wasn't computable).
    """
    reading: FrameReading = {}

    # Body lean
    if metrics.body_angle_deg is not None:
        reading["body"] = classify_body_lean(
            metrics.body_angle_deg, thresholds.T_body
        )
    else:
        reading["body"] = None

    # Head direction
    if metrics.head_local_x is not None:
        reading["head"] = classify_head_direction(
            metrics.head_local_x, thresholds.T_head
        )
    else:
        reading["head"] = None

    # Hand movement (velocity primary, position fallback)
    reading["hand"] = classify_limb_movement(
        metrics.hand_v_left, metrics.hand_v_right, thresholds.T_hand
    )
    if reading["hand"] is None:
        reading["hand"] = classify_limb_position(
            metrics.wrist_left_local_x, metrics.wrist_right_local_x,
        )

    # Leg movement (velocity primary, position fallback)
    reading["leg"] = classify_limb_movement(
        metrics.leg_v_left, metrics.leg_v_right, thresholds.T_leg
    )
    if reading["leg"] is None:
        reading["leg"] = classify_limb_position(
            metrics.ankle_left_local_x, metrics.ankle_right_local_x,
        )

    return reading

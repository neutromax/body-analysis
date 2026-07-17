"""
Auto Annotator — Geometry Module
=================================

Pure vector math for body-language pose analysis.

This module contains ONLY mathematical operations — no classification
thresholds, no -1/0/1 labels, no config imports.  Every function takes
numbers in and returns numbers out, making them independently testable
and reusable.

Coordinate conventions (MediaPipe normalized image space):
    - x increases rightward   (0.0 = left edge, 1.0 = right edge)
    - y increases DOWNWARD     (0.0 = top edge, 1.0 = bottom edge)
    - A person standing upright has shoulders ABOVE hips,
      so shoulder_y < hip_y in raw coordinates.

Key concepts:
    - C_s  (shoulder center)  — origin for head/hand measurements
    - C_h  (hip center)       — base of the torso
    - S    (shoulder width)   — the normalization scale factor
    - u    (body "up" axis)   — unit vector from hips toward shoulders
    - r    (body "right" axis) — perpendicular to u, pointing body-right
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# Type alias for a 2D point or vector.
Vec2 = Tuple[float, float]


# ───────────────────────────────────────────────────────────────────────
# Primitive operations
# ───────────────────────────────────────────────────────────────────────

def midpoint(a: Vec2, b: Vec2) -> Vec2:
    """
    Return the midpoint of two 2D points.

    Used to compute shoulder center C_s and hip center C_h from left/right
    landmark pairs.
    """
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def euclidean_dist(a: Vec2, b: Vec2) -> float:
    """
    Euclidean distance between two 2D points.

    Used to compute shoulder width S, which normalizes all distance and
    velocity measurements so the same physical movement produces the same
    number regardless of camera distance or body size.
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.sqrt(dx * dx + dy * dy)


def dot(a: Vec2, b: Vec2) -> float:
    """
    Dot product of two 2D vectors.

    Used to project a point onto the body's up or right axis, yielding
    the body-relative local_x or local_y coordinate.
    """
    return a[0] * b[0] + a[1] * b[1]


def unit_vector(v: Vec2) -> Vec2:
    """
    Normalize a 2D vector to unit length.

    Returns (0, 0) if the input vector has zero length (degenerate case
    when shoulders and hips coincide — should not happen with a valid
    pose, but guarding against division by zero).
    """
    length = math.sqrt(v[0] * v[0] + v[1] * v[1])
    if length < 1e-9:
        return (0.0, 0.0)
    return (v[0] / length, v[1] / length)


# ───────────────────────────────────────────────────────────────────────
# Body reference frame
# ───────────────────────────────────────────────────────────────────────

def body_reference_frame(
    left_shoulder: Vec2,
    right_shoulder: Vec2,
    left_hip: Vec2,
    right_hip: Vec2,
) -> Tuple[Vec2, Vec2, float, Vec2, Vec2]:
    """
    Compute the body-centric coordinate frame from four torso landmarks.

    Returns:
        C_s:  Shoulder center — midpoint of LEFT_SHOULDER and RIGHT_SHOULDER.
        C_h:  Hip center — midpoint of LEFT_HIP and RIGHT_HIP.
        S:    Shoulder width (Euclidean distance between shoulders).
              Used as the scale factor to normalize all measurements.
        u:    Unit vector pointing "up" along the torso (from C_h toward
              C_s).  When the person is upright, u ≈ (0, -1).
        r:    Unit vector pointing "right" in the body's own frame,
              perpendicular to u.  Computed as the 90° clockwise rotation
              of u:  r = (-u_y, u_x).
              When the person is upright, r ≈ (1, 0)  — screen-right.

    Why we need this:
        Measuring "left/right" in raw image coordinates breaks when the
        person leans.  By projecting landmarks onto (u, r) we measure
        lateral offset relative to the torso itself, which is robust to
        lean, camera angle, and body size.
    """
    C_s = midpoint(left_shoulder, right_shoulder)
    C_h = midpoint(left_hip, right_hip)
    S = euclidean_dist(left_shoulder, right_shoulder)

    # Torso axis: vector from hip center UP toward shoulder center.
    v_body = (C_s[0] - C_h[0], C_s[1] - C_h[1])
    u = unit_vector(v_body)

    # Right axis: 90° clockwise rotation of u.
    # If u = (ux, uy), then rotating 90° CW gives (-uy, ux).
    # When upright u ≈ (0, -1)  →  r = (1, 0)  ✓
    r = (-u[1], u[0])

    return C_s, C_h, S, u, r


# ───────────────────────────────────────────────────────────────────────
# Projections into the body frame
# ───────────────────────────────────────────────────────────────────────

def torso_local_position(
    point: Vec2,
    C_s: Vec2,
    u: Vec2,
    r: Vec2,
    S: float,
) -> Tuple[float, float]:
    """
    Project a point into the body's local coordinate system.

    Args:
        point:  The landmark to project (e.g. NOSE, LEFT_WRIST).
        C_s:    Shoulder center (the origin of the body frame).
        u:      Body "up" unit vector.
        r:      Body "right" unit vector.
        S:      Shoulder width (scale factor).

    Returns:
        (local_x, local_y) where:
            local_x = dot(point - C_s, r) / S
                Positive → point is to the body's right.
                Negative → point is to the body's left.
            local_y = dot(point - C_s, u) / S
                Positive → point is above shoulder center (along body up).
                Negative → point is below shoulder center.

    Both values are normalized by shoulder width, so they're dimensionless
    and comparable across frames and individuals.
    """
    if S < 1e-9:
        return (0.0, 0.0)

    w = (point[0] - C_s[0], point[1] - C_s[1])
    local_x = dot(w, r) / S
    local_y = dot(w, u) / S
    return (local_x, local_y)


# ───────────────────────────────────────────────────────────────────────
# Body lean angle
# ───────────────────────────────────────────────────────────────────────

def torso_lean_angle(
    left_shoulder: Vec2,
    right_shoulder: Vec2,
    left_hip: Vec2,
    right_hip: Vec2,
) -> float:
    """
    Signed angle (in degrees) between the torso axis and true vertical.

    Derivation:
        Let V = (0, -1) be true vertical (upward in screen space).
        Let v_body = C_s - C_h  (torso axis from hips to shoulders).

        The signed angle from V to v_body is:
            theta = atan2(cross(V, v_body), dot(V, v_body))

        where:
            cross(V, v_body) = 0 * v_body_y - (-1) * v_body_x = v_body_x
            dot(V, v_body)   = 0 * v_body_x + (-1) * v_body_y = -v_body_y

        So:  theta = atan2(v_body_x, -v_body_y)

    Sign convention:
        Positive → torso leans RIGHT  (v_body tilts toward screen-right)
        Negative → torso leans LEFT   (v_body tilts toward screen-left)
        Zero     → perfectly upright

    Returns:
        Angle in degrees.
    """
    C_s = midpoint(left_shoulder, right_shoulder)
    C_h = midpoint(left_hip, right_hip)

    v_body_x = C_s[0] - C_h[0]
    v_body_y = C_s[1] - C_h[1]

    theta_rad = math.atan2(v_body_x, -v_body_y)
    return math.degrees(theta_rad)


# ───────────────────────────────────────────────────────────────────────
# Head lateral offset (body-relative)
# ───────────────────────────────────────────────────────────────────────

def head_lateral_offset(
    head_point: Vec2,
    C_s: Vec2,
    r: Vec2,
    S: float,
) -> float:
    """
    Lateral offset of the head landmark from shoulder center, measured
    along the body's own right axis.

    This is the local_x component of the head's position in the body
    frame.  Using the body frame (r) instead of raw image x means that
    if the person leans right, the nose doesn't falsely register as
    "turned right."

    Args:
        head_point:  The head landmark (NOSE or ear midpoint).
        C_s:         Shoulder center.
        r:           Body "right" unit vector.
        S:           Shoulder width.

    Returns:
        local_x_head (float):
            Positive → head is to the body's right.
            Negative → head is to the body's left.
    """
    if S < 1e-9:
        return 0.0

    w = (head_point[0] - C_s[0], head_point[1] - C_s[1])
    return dot(w, r) / S


# ───────────────────────────────────────────────────────────────────────
# Limb lateral velocity (body-relative, between consecutive frames)
# ───────────────────────────────────────────────────────────────────────

def limb_lateral_velocity(
    local_x_now: float,
    local_x_prev: float,
) -> float:
    """
    Lateral displacement of a limb landmark between two consecutive
    1-second samples, measured in body-relative coordinates.

    This is the VELOCITY metric used for Hand Movement and Leg Movement.
    It measures how much the limb moved laterally (in the body's own
    frame) over the last 1-second step.

    Args:
        local_x_now:   Current second's body-relative lateral position.
        local_x_prev:  Previous second's body-relative lateral position.

    Returns:
        v (float):
            Positive → limb moved to the body's RIGHT this second.
            Negative → limb moved to the body's LEFT this second.
            Zero     → no lateral movement.
    """
    return local_x_now - local_x_prev

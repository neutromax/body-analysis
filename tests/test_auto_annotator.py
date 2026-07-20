"""
Tests for the Auto Annotator components: geometry, classifier, calibration, and aggregator.
"""

import sys
import os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "auto_annotator"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import geometry
import feature_classifier
import calibration
import aggregator
from excel_writer import ExcelWriter
from tempfile import TemporaryDirectory


class TestGeometry:
    """Tests for pure math helper functions in geometry.py."""

    def test_midpoint(self):
        a = (1.0, 2.0)
        b = (3.0, 4.0)
        assert geometry.midpoint(a, b) == (2.0, 3.0)

    def test_euclidean_dist(self):
        a = (0.0, 0.0)
        b = (3.0, 4.0)
        assert math.isclose(geometry.euclidean_dist(a, b), 5.0)

    def test_unit_vector(self):
        v = (3.0, 4.0)
        u = geometry.unit_vector(v)
        assert math.isclose(u[0], 0.6)
        assert math.isclose(u[1], 0.8)
        # Degenerate case
        assert geometry.unit_vector((0.0, 0.0)) == (0.0, 0.0)

    def test_body_reference_frame(self):
        ls = (0.3, 0.2)
        rs = (0.7, 0.2)
        lh = (0.3, 0.8)
        rh = (0.7, 0.8)
        C_s, C_h, S, u, r = geometry.body_reference_frame(ls, rs, lh, rh)

        assert C_s == (0.5, 0.2)
        assert C_h == (0.5, 0.8)
        assert math.isclose(S, 0.4)

        # Torso vector goes from (0.5, 0.8) to (0.5, 0.2), which is (0.0, -0.6)
        # Unit vector should be (0.0, -1.0) (upwards in screen coords)
        assert math.isclose(u[0], 0.0)
        assert math.isclose(u[1], -1.0)

        # Perpendicular should point CW: r = (-u_y, u_x) = (1.0, 0.0) (rightward)
        assert math.isclose(r[0], 1.0)
        assert math.isclose(r[1], 0.0)

    def test_torso_local_position(self):
        C_s = (0.5, 0.2)
        u = (0.0, -1.0)
        r = (1.0, 0.0)
        S = 0.4

        # Point at (0.7, 0.1)
        # w = point - C_s = (0.2, -0.1)
        # local_x = dot(w, r) / S = 0.2 / 0.4 = 0.5
        # local_y = dot(w, u) / S = (-0.2*0 + -0.1*-1) / 0.4 = 0.1 / 0.4 = 0.25
        lx, ly = geometry.torso_local_position((0.7, 0.1), C_s, u, r, S)
        assert math.isclose(lx, 0.5)
        assert math.isclose(ly, 0.25)

    def test_torso_lean_angle(self):
        # Perfectly upright
        ls, rs = (0.3, 0.2), (0.7, 0.2)
        lh, rh = (0.3, 0.8), (0.7, 0.8)
        assert math.isclose(geometry.torso_lean_angle(ls, rs, lh, rh), 0.0, abs_tol=1e-5)

        # Leaning to screen-right by 45 degrees
        # C_h = (0.5, 0.8), C_s shifted right
        # v_body = C_s - C_h = (0.6, -0.6)
        # atan2(0.6, 0.6) = 45 degrees
        ls, rs = (0.9, 0.2), (1.3, 0.2)
        lh, rh = (0.3, 0.8), (0.7, 0.8)
        assert math.isclose(geometry.torso_lean_angle(ls, rs, lh, rh), 45.0)


class TestFeatureClassifier:
    """Tests for classification logic in feature_classifier.py."""

    def test_classify_body_lean(self):
        # T_body = 8.0
        assert feature_classifier.classify_body_lean(10.0, 8.0) == 1
        assert feature_classifier.classify_body_lean(-10.0, 8.0) == -1
        assert feature_classifier.classify_body_lean(5.0, 8.0) == 0

    def test_classify_head_direction(self):
        # T_head = 0.04
        assert feature_classifier.classify_head_direction(0.05, 0.04) == 1
        assert feature_classifier.classify_head_direction(-0.05, 0.04) == -1
        assert feature_classifier.classify_head_direction(0.02, 0.04) == 0

    def test_classify_limb_movement(self):
        # Both None
        assert feature_classifier.classify_limb_movement(None, None, 0.03) is None

        # Left only, moving right
        assert feature_classifier.classify_limb_movement(0.05, None, 0.03) == 1
        # Left only, neutral
        assert feature_classifier.classify_limb_movement(-0.01, None, 0.03) == 0

        # Right only, moving left
        assert feature_classifier.classify_limb_movement(None, -0.04, 0.03) == -1

        # Both active, left dominant moving right
        assert feature_classifier.classify_limb_movement(0.05, -0.02, 0.03) == 1
        # Both active, right dominant moving left
        assert feature_classifier.classify_limb_movement(0.01, -0.05, 0.03) == -1

    def test_classify_limb_position(self):
        # Both None
        assert feature_classifier.classify_limb_position(None, None, 0.15) is None

        # Left only, displaced right
        assert feature_classifier.classify_limb_position(0.20, None, 0.15) == 1
        # Left only, neutral
        assert feature_classifier.classify_limb_position(0.10, None, 0.15) == 0

        # Right only, displaced left
        assert feature_classifier.classify_limb_position(None, -0.20, 0.15) == -1

        # Both active, left dominant displaced right
        assert feature_classifier.classify_limb_position(0.25, -0.10, 0.15) == 1
        # Both active, right dominant displaced left
        assert feature_classifier.classify_limb_position(0.05, -0.30, 0.15) == -1


class TestCalibration:
    """Tests for threshold calibration in calibration.py."""

    def test_mean_and_std(self):
        mu, sigma = calibration._mean_and_std([1.0, 2.0, 3.0])
        assert math.isclose(mu, 2.0)
        assert math.isclose(sigma, math.sqrt(2.0/3.0))

    def test_compute_threshold_fallback(self):
        # Too few readings
        assert calibration._compute_threshold([1.0], 1.0, 5.0, "test") == 5.0

    def test_compute_threshold_math(self):
        # Values: 1.0, 2.0, 3.0. mu=2.0, sigma=0.81649658
        # T = |mu| + k*sigma = 2.0 + 1.0 * 0.81649658 = 2.81649658
        val = calibration._compute_threshold([1.0, 2.0, 3.0], 1.0, 5.0, "test")
        assert math.isclose(val, 2.0 + math.sqrt(2.0/3.0))

    def test_calibrator_is_complete(self):
        cal = calibration.Calibrator(duration=3, k=1.0)
        assert not cal.is_complete
        assert cal.readings_collected == 0

        # Frame 1
        metrics = feature_classifier.FrameMetrics(
            body_angle_deg=1.0, head_local_x=0.01,
            hand_v_left=0.01, hand_v_right=0.005,
            leg_v_left=0.002, leg_v_right=0.001
        )
        cal.add_reading(metrics)
        assert not cal.is_complete
        assert cal.readings_collected == 1

        # Frame 2
        cal.add_reading(metrics)
        # Frame 3
        cal.add_reading(metrics)

        assert cal.is_complete
        assert cal.readings_collected == 3

        thresholds = cal.compute_thresholds()
        assert thresholds.calibrated is True
        # Since all readings were identical, sigma=0, so threshold is |mu| = 1.0 for body
        assert math.isclose(thresholds.T_body, 1.0)


class TestAggregator:
    """Tests for majority voting and tie breaking in aggregator.py."""

    def test_majority_vote_clear(self):
        # Clean majority
        assert aggregator.majority_vote([-1, -1, 0, -1, 1, -1, 0, -1, -1, 0]) == -1

    def test_majority_vote_all_na(self):
        assert aggregator.majority_vote([None, None, None]) is None

    def test_majority_vote_tie_break(self):
        # Tie between -1 and 1, last non-NA is 1
        assert aggregator.majority_vote([-1, -1, 1, 1, None], tie_break_rule="most_recent") == 1
        # Tie between -1 and 1, last non-NA is -1
        assert aggregator.majority_vote([1, 1, -1, -1, None], tie_break_rule="most_recent") == -1

    def test_aggregate_window(self):
        readings = [
            {"hand": 1, "leg": 0, "head": -1, "body": 1},
            {"hand": 1, "leg": 0, "head": 0, "body": 1},
            {"hand": None, "leg": None, "head": None, "body": None},
        ]
        result = aggregator.aggregate_window(readings)
        assert result["hand"] == 1
        assert result["leg"] == 0
        assert result["head"] == 0   # Tie (-1 vs 0): most_recent non-NA is 0 (2nd reading)
        assert result["body"] == 1

    def test_compute_frame_metrics_fallback(self):
        # Shoulders visible, hips missing
        # Left shoulder at (0.7, 0.2), Right shoulder at (0.3, 0.2), Nose at (0.5, 0.1)
        # Wrist at (0.6, 0.4)
        landmarks = {
            "LEFT_SHOULDER": (0.7, 0.2, 0.9),
            "RIGHT_SHOULDER": (0.3, 0.2, 0.9),
            "NOSE": (0.5, 0.1, 0.9),
            "LEFT_WRIST": (0.6, 0.4, 0.9),
            "RIGHT_WRIST": (0.4, 0.4, 0.9),
        }
        
        # Hips are not in the landmarks dictionary at all, or have visibility < 0.3
        metrics = feature_classifier.compute_frame_metrics(landmarks, None)
        
        # Check that we computed body lean, head local x, and wrist positions
        assert metrics.body_angle_deg is not None
        assert math.isclose(metrics.body_angle_deg, 0.0, abs_tol=1e-5) # Upright shoulders
        assert metrics.head_local_x is not None
        assert math.isclose(metrics.head_local_x, 0.0, abs_tol=1e-5) # Nose centered
        assert metrics.wrist_left_local_x is not None
        assert metrics.wrist_right_local_x is not None
        
        # Legs should remain None because legs always require actual hips and ankles
        assert metrics.ankle_left_local_x is None
        assert metrics.leg_v_left is None


class TestExcelWriter:
    def test_corrupted_excel_fallback(self):
        with TemporaryDirectory() as tmpdir:
            corrupted_file = os.path.join(tmpdir, "corrupted.xlsx")
            # Create a file that is not a valid zip archive (just write plain text to it)
            with open(corrupted_file, "w") as f:
                f.write("corrupted excel content")
            
            # Initializing ExcelWriter on it should fall back to creating a fresh one
            writer = ExcelWriter(corrupted_file)
            assert writer._wb is not None
            
            # Verify the headers are present in the fresh workbook
            ws = writer._ws
            assert ws.cell(row=1, column=1).value == "Video Link"
            assert ws.cell(row=1, column=2).value == "Time Window"


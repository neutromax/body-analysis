"""
Auto Annotator — Configuration
===============================

Single source of truth for every constant in the automated annotation system.
Renamed to auto_config.py to avoid clashes with parent project config.py.
"""

# ═══════════════════════════════════════════════════════════════════════
# 1. TIMING
# ═══════════════════════════════════════════════════════════════════════

# How often (in seconds) we capture a frame and classify it.
OBSERVATION_INTERVAL: int = 1

# How many seconds of readings are aggregated into one output row.
WINDOW_DURATION: int = 10

# Derived: observations per window (10 readings → 1 row).
OBSERVATIONS_PER_WINDOW: int = WINDOW_DURATION // OBSERVATION_INTERVAL

# Maximum seconds to wait for the YouTube player to load and become ready.
PLAYER_TIMEOUT: int = 20



# ═══════════════════════════════════════════════════════════════════════
# 2. CALIBRATION
# ═══════════════════════════════════════════════════════════════════════

# Number of seconds at the start of each video used as the neutral
# baseline for threshold calibration (K in the spec).
CALIBRATION_DURATION: int = 5

# How many standard deviations of natural resting variation a reading
# must exceed to be classified as intentional movement (k in the spec).
SIGMA_MULTIPLIER: float = 1.0

# Fallback thresholds — used ONLY when calibration fails.
FALLBACK_T_BODY: float = 8.0    # degrees of torso lean from vertical
FALLBACK_T_HEAD: float = 0.04   # body-frame lateral offset (normalized)
FALLBACK_T_HAND: float = 0.03   # body-frame lateral velocity (normalized)
FALLBACK_T_LEG: float = 0.02    # body-frame lateral velocity (normalized)


# ═══════════════════════════════════════════════════════════════════════
# 3. POSE (MediaPipe Landmark Indices)
# ═══════════════════════════════════════════════════════════════════════

# Minimum MediaPipe visibility score (0.0–1.0).
VISIBILITY_THRESHOLD: float = 0.3

# MediaPipe Pose landmark indices (from the 33-landmark model).
LANDMARK = {
    "NOSE":             0,
    "LEFT_EAR":         7,
    "RIGHT_EAR":        8,
    "LEFT_SHOULDER":    11,
    "RIGHT_SHOULDER":   12,
    "LEFT_ELBOW":       13,
    "RIGHT_ELBOW":      14,
    "LEFT_WRIST":       15,
    "RIGHT_WRIST":      16,
    "LEFT_HIP":         23,
    "RIGHT_HIP":        24,
    "LEFT_KNEE":        25,
    "RIGHT_KNEE":       26,
    "LEFT_ANKLE":       27,
    "RIGHT_ANKLE":      28,
}

# Which landmark to use for head direction measurement.
HEAD_LANDMARK_MODE: str = "NOSE"


# ═══════════════════════════════════════════════════════════════════════
# 4. FEATURES
# ═══════════════════════════════════════════════════════════════════════

# The four body-language features we classify.
FEATURES: list = ["hand", "leg", "head", "body"]

# Encoding for each feature value.
ENCODING: dict = {
    -1: "Left",
     0: "Center",
     1: "Right",
}

# Tie-break rule for majority vote.
TIE_BREAK_RULE: str = "most_recent"


# ═══════════════════════════════════════════════════════════════════════
# 5. EXCEL OUTPUT
# ═══════════════════════════════════════════════════════════════════════

# Column names in the output Excel file.
EXCEL_COLUMNS: list = [
    "Video Link",
    "Time Window",
    "Hand",
    "Leg",
    "Head",
    "Body",
]


# ═══════════════════════════════════════════════════════════════════════
# 6. GUI SETTINGS
# ═══════════════════════════════════════════════════════════════════════

APP_TITLE: str = "Auto Body Language Annotator"
APP_WIDTH: int = 520
APP_HEIGHT: int = 600

# pywebview player window dimensions
PLAYER_WIDTH: int = 854
PLAYER_HEIGHT: int = 540

# Dark theme color palette
COLORS: dict = {
    "bg":           "#1a1b26",
    "bg_light":     "#24283b",
    "bg_card":      "#2f3348",
    "text":         "#c0caf5",
    "text_dim":     "#565f89",
    "accent":       "#7aa2f7",
    "accent_hover": "#89b4fa",
    "success":      "#9ece6a",
    "warning":      "#e0af68",
    "error":        "#f7768e",
    "left_color":   "#f7768e",
    "center_color": "#7aa2f7",
    "right_color":  "#9ece6a",
    "border":       "#3b4261",
}

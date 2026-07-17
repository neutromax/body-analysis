"""
Auto Annotator — Aggregator
=============================

Majority vote over each 10-second window's one-second readings.

Rules (Section 6.8):
    1. For each feature, count how many valid (non-NA) readings are
       -1, 0, and 1.
    2. The value with the highest count wins.
    3. On a TIE: use the value from the MOST RECENT non-NA second in
       that window (deterministic, documented).
    4. If ALL readings are NA: the window's value is NA (None).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional

from auto_config import FEATURES, TIE_BREAK_RULE

logger = logging.getLogger(__name__)

# Type alias: one second's reading — feature name → (-1, 0, 1, or None).
FrameReading = Dict[str, Optional[int]]

# Type alias: one window's aggregated result — feature name → (-1, 0, 1, or None).
WindowResult = Dict[str, Optional[int]]


def majority_vote(
    values: List[Optional[int]],
    tie_break_rule: str = TIE_BREAK_RULE,
) -> Optional[int]:
    """
    Compute the majority value from a list of per-second readings.

    Args:
        values:          List of int values (-1, 0, 1) or None (NA).
                         NA values are excluded from the vote.
        tie_break_rule:  How to resolve ties.
                         "most_recent" → use the last non-NA value in the list.

    Returns:
        The winning value (-1, 0, or 1), or None if every reading is NA.

    Examples:
        >>> majority_vote([-1, -1, 0, -1, 1, -1, 0, -1, -1, 0])
        -1
        >>> majority_vote([-1, -1, -1, 1, 1, 1, 0, 0, 0, None])
        0   # three-way tie → most_recent non-NA is 0
        >>> majority_vote([None, None, None])
        None
    """
    # Filter out NA values.
    valid = [v for v in values if v is not None]

    if not valid:
        return None   # Every reading was NA.

    counter = Counter(valid)
    most_common = counter.most_common()

    # Check for ties: top value(s) share the same count.
    max_count = most_common[0][1]
    tied = [val for val, cnt in most_common if cnt == max_count]

    if len(tied) == 1:
        return tied[0]   # Clear winner.

    # Tie-break: use the most recent non-NA value.
    if tie_break_rule == "most_recent":
        # Walk backward through the original list to find the last non-NA.
        for v in reversed(values):
            if v is not None and v in tied:
                return v

    # Fallback (shouldn't reach here, but be safe): return 0.
    return 0


def aggregate_window(readings: List[FrameReading]) -> WindowResult:
    """
    Aggregate a full 10-second window's per-second readings into one
    result row.

    For each feature, collects all readings across the window and runs
    majority_vote.

    Args:
        readings:  List of up to 10 FrameReading dicts (one per second).

    Returns:
        Dict with one value per feature:
            {"hand": 1, "leg": 0, "head": -1, "body": None}
    """
    result: WindowResult = {}

    for feature in FEATURES:
        feature_values = [r.get(feature) for r in readings]
        result[feature] = majority_vote(feature_values)

    return result
